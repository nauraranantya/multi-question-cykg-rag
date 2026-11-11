# src/agents/mcp_rdf_agent.py
"""
`import mcp_use` is deliberately NOT at module level (see git history/
PROTOTYPE.md for the incident this fixes): mcp_use/__init__.py eagerly
imports mcp_use.observability, which initializes Laminar/Langfuse tracing
integrations at import time, and that import has been observed to hang
indefinitely (blocked in a worker-thread `read()`, confirmed with `sample`
on the stuck process) even with no Laminar/Langfuse API keys configured --
i.e. even the branch that's supposed to no-op can stall. Since
src/graph/workflow.py imports this module eagerly (`from
src.agents.mcp_rdf_agent import run_mcp_agent`), an eager `import mcp_use`
here would take down the entire investigation service's startup (and
src/run.py's / eval's imports) over a CSKG-lookup dependency that most
questions never even touch. Deferred to inside get_mcp_client()/
run_mcp_agent() instead -- a hang there now only affects the one CSKG
question that triggers it (already wrapped in NODE_CALL_TIMEOUT_SECONDS by
src/graph/workflow.py's dispatch_retrieval_node), not process startup.
"""
import asyncio
import contextvars
import hashlib
import json
import re
import time
import os
import logging
import time
from src.agents.cskg_health import CskgDependencyError, load_server_config, run_checks_async
from src.config.settings import llm

logger = logging.getLogger(__name__)

# --- Konfigurasi dan Inisialisasi Agen MCP ---

_mcp_client = None
MCP_MAX_STEPS = max(1, int(os.environ.get("CSKG_MCP_MAX_STEPS", "5")))
MCP_MAX_RESULT_ROWS = max(1, int(os.environ.get("CSKG_MAX_RESULT_ROWS", "10")))

def get_mcp_client():
    """Inisialisasi dan mengembalikan MCPClient (hanya sekali).

    The server definition comes from cskg_health.load_server_config(), not
    straight from the file: that resolves ${CURRENT_PYTHON}/${PROJECT_ROOT}
    (and a CSKG_PYTHON override) so browser_mcp.json no longer hard-codes
    one machine's virtualenv -- the pilot's CSKG failure was exactly a
    hard-coded interpreter whose environment had been offloaded. It is the
    same definition the independent health check launches."""
    global _mcp_client
    if _mcp_client is None:
        config = load_server_config()
        # Must be set before `import mcp_use` -- its observability
        # submodules read these via os.getenv at their own import time
        # (see module docstring above), not lazily.
        os.environ["MCP_USE_ANONYMIZED_TELEMETRY"] = "false"
        os.environ.setdefault("MCP_USE_LAMINAR", "false")
        os.environ.setdefault("MCP_USE_LANGFUSE", "false")
        from mcp_use import MCPClient
        _mcp_client = MCPClient.from_dict(config)
    return _mcp_client


# --- Readiness: fail fast, and say why ---
# Before the first CSKG question in a process (and again after a failure,
# once the cool-down passes) the server is probed with the independent
# check in cskg_health.py. A broken dependency then surfaces within one
# retrieval deadline as a CskgDependencyError naming the failing stage,
# instead of each question waiting out a 60 s timeout and reporting ''.
READINESS_FAILURE_TTL_SECONDS = float(os.environ.get("CSKG_READINESS_RETRY_SECONDS", "120"))
_readiness = {"report": None, "at": 0.0}
_readiness_locks: dict = {}


async def ensure_cskg_ready() -> dict:
    loop = asyncio.get_running_loop()
    lock = _readiness_locks.setdefault(id(loop), asyncio.Lock())
    async with lock:
        report = _readiness["report"]
        fresh = report is not None and (
            report["status"] == "ok" or time.monotonic() - _readiness["at"] < READINESS_FAILURE_TTL_SECONDS)
        if not fresh:
            report = await run_checks_async(deep=False)
            _readiness.update(report=report, at=time.monotonic())
            log = logger.info if report["status"] == "ok" else logger.error
            log(f"[cskg readiness] {report['summary']}")
    if report["status"] != "ok":
        raise CskgDependencyError(report["summary"], stage=report.get("failed_stage"), report=report)
    return report


def last_readiness_report():
    """Most recent readiness probe result for this process, or None."""
    return _readiness["report"]


def _ensure_step_budget_not_exhausted(steps_taken: int, max_steps: int) -> None:
    """Reject a returned-but-truncated answer when the tool loop hit its cap."""
    if steps_taken >= max_steps:
        raise CskgDependencyError(
            f"CSKG agent exhausted its {max_steps}-step tool budget before confirming a complete answer",
            stage="agent_step_limit",
        )


_TECHNIQUE_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_CVE_ID_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
_CVE_TOOLS = ["get_cve_by_id", "search_cves_for_product", "search_cves_by_keyword", "get_references_for_cve"]


def _is_cve_question(question: str) -> bool:
    text = question.lower()
    return bool(_CVE_ID_RE.search(question)) or 'cve' in text or 'vulnerab' in text or 'cvss' in text


def _is_technique_mitigation_question(question: str) -> bool:
    text = question.lower()
    if _is_cve_question(question):
        return False
    return 'mitigation' in text or ('att&ck' in text and 'technique' in text)


def _allowed_tools(question: str):
    """Which tools a question may use, or None for the unrestricted agent.
    Each focused question type gets its purpose-built tool(s) only, so the
    agent cannot spend its step budget on broad searches or on
    text_to_sparql's separate OpenAI API call."""
    if _is_cve_question(question):
        return _CVE_TOOLS
    if _is_technique_mitigation_question(question):
        if _TECHNIQUE_ID_RE.search(question):
            return ["get_mitigations_for_technique_ids"]
        return ["get_technique_mitigations_by_keyword"]
    return None


def _disallow_tools_except(tool_names, allowed) -> list[str]:
    allowed = {allowed} if isinstance(allowed, str) else set(allowed)
    return [name for name in tool_names if name not in allowed]

# Exact marker MAX_STEPS-exhausted/nothing-found responses must start
# with -- src/graph/workflow.py::_context_richness matches on this (plus a
# phrase-based heuristic fallback for whenever the model doesn't follow
# this instruction exactly) to tell "found nothing" apart from "found
# something," instead of the two being indistinguishable natural-language
# prose. Without this, a CSKG "I couldn't find anything" answer was being
# counted as substantive evidence, inflating the confidence cap
# _context_richness/_cap_confidence exist to enforce.
NO_DATA_MARKER = "NO_RELEVANT_DATA_FOUND"

strict_system_prompt = f"""
You are a specialized cybersecurity assistant. You MUST answer questions ONLY by using the provided tools. Your only source of information is the knowledge graph accessed via tools.

Keep retrieval bounded: use the fewest tool calls that can answer the question, prefer one precise lookup, and do not request exhaustive lists. If a call times out, do not repeat the same query; make at most one narrower fallback. The tool loop is limited to {MCP_MAX_STEPS} steps, and each result is capped at {MCP_MAX_RESULT_ROWS} rows.

When techniques are already known by ID (e.g. T1046, T1505.003) and their mitigations are asked for, make ONE call to get_mitigations_for_technique_ids with all the IDs; it returns each technique's name and mitigations.
When the question asks which CVEs affect a named software product or version, use search_cves_for_product (product, and version if known); use get_cve_by_id for a known CVE ID. Report each CVE ID with its CVSS v3 base score and severity exactly as returned, say that matching is on description text so version ranges may be missed, and never claim that a CVE was exploited in this environment.
When asked to map observed behavior to ATT&CK techniques and/or recommend mitigations, use get_technique_mitigations_by_keyword with a list of up to three concise phrases drawn from the observed signals (for example, the relevant service behavior, a scan/banner phrase, and a DNS query type). Make one call; only if it returns no results, make one fallback call with alternate short phrases. It returns bounded technique IDs/titles and linked mitigation titles. Never use text_to_sparql for these tasks: it makes a separate hard-coded OpenAI GPT-4o API call, bypassing the configured OpenRouter model/provider. Do not replace graph evidence with memory, or search technique and mitigation tools separately.

This knowledge base ONLY contains public reference data: MITRE ATT&CK techniques/tactics/groups/software/mitigations,
and CVE/CVSS records. It has NO IP addresses, hostnames, domains, live threat-intelligence feeds, IOC reputation data,
or any organization-specific data whatsoever. A question that asks you to correlate, attribute, or look up a
*specific* IP address, hostname, or piece of an organization's own environment against threat intelligence cannot be
answered here at all, no matter how the tools are used -- recognize this immediately from the question itself,
without attempting tool calls first, and respond per rule 6 below.

Most tools that take a technique/group/mitigation/software name (get_mitigations_for_technique,
get_techniques_by_tactic, get_techniques_used_by_group, etc.) match against that entity's TITLE TEXT
(e.g. "Network Service Discovery"), NOT its MITRE ID (e.g. "T1046") -- passing a bare ID to one of those
tools will silently return nothing, every time, no matter how many times you retry it, because the ID
never appears as a substring of the title. If a question names a technique only by ID: first call
get_technique_by_id to resolve it to its real name, THEN use that name with a name-matching tool -- OR use
text_to_sparql to write a single query that filters on the ID directly (FILTER(STRENDS(STR(?technique),
"/T1046"))), which sidesteps the name-matching requirement entirely.

Your thought process MUST be:
    1.  **Analyze the user's question** to understand the core intent.
    2.  **Select the best tool.**.
    3.  **Execute the tool.** Prefer one precise lookup. If it fails, do not repeat the same query; use at most one narrower `text_to_sparql` fallback.
    4.  **Analyze the result.**
        - If the result is a **validation error** (like a Pydantic error), it means you provided the wrong arguments to the tool. Read the error message carefully. DO NOT use the same tool with the exact same arguments again. Correct the arguments and retry. For tools requiring `ctx`, DO NOT provide a value for it; the system handles it.
        - If the result is **empty or "not found"**, the information may not exist, or your query was too narrow. Try rephrasing your input for the tool, perhaps using a broader term.
        - If you are stuck in a loop of failures, only then you must state that you could not find the information.
    5.  **NEVER provide an answer from memory.** All answers must be based on tool results.
    6.  **If the question needs data this knowledge base doesn't have at all** (per the scope note above), OR **if,
        after reasonable attempts, no relevant information was found**, your final answer MUST begin with the exact
        marker "{NO_DATA_MARKER}" (nothing before it), followed by one short sentence saying why (out of scope, or
        what you tried). This applies even when you go on to offer a related but different reformulation of the
        question -- the marker reflects that the ORIGINAL question wasn't answered, regardless of what else you add
        after it. A downstream check relies on this exact marker to tell "found nothing" apart from "found
        something" -- do not paraphrase it, and do not include it if you DID find relevant information answering the
        question as actually asked. If a tool returned rows that answer PART of the question (for example the
        mitigations but not the description), do NOT use the marker: report what was found and add one sentence
        naming the part that could not be retrieved.
"""

# --- Per-question trace of knowledge-graph tool calls ---------------------
# Recorded with each question result, so an evaluation can tell which tools
# were used, with which arguments, and what they returned. A call's start
# time and duration also join it to the server's per-query telemetry
# (cskg_events: query hash, cache hit/miss, failures).
_TRACE = contextvars.ContextVar("cskg_trace", default=None)
_TRACE_PREVIEW_CHARS = 400
_TRACE_ARGUMENT_CHARS = 500


def start_trace() -> dict:
    """Begin collecting a trace for the next run_mcp_agent call in this
    context (asyncio tasks created afterwards share the same dict)."""
    trace = {"allowed_tools": None, "max_steps": None, "steps": None, "tools_used": [], "calls": []}
    _TRACE.set(trace)
    return trace


def _result_text(result) -> str:
    parts = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        parts.append(text if isinstance(text, str) else str(item))
    return "\n".join(parts) if parts else str(result)


def _instrument_connector(connector) -> None:
    """Wrap one MCP connector's call_tool once. The tool adapters look the
    method up on every call, so the wrapper sees every agent tool call."""
    if getattr(connector, "_cskg_trace_wrapped", False):
        return
    original = connector.call_tool

    async def call_tool(name, arguments, *args, **kwargs):
        trace = _TRACE.get()
        started = time.time()
        record = {"tool": name, "arguments": json.dumps(arguments, default=str)[:_TRACE_ARGUMENT_CHARS],
                  "started": round(started, 3)}
        try:
            result = await original(name, arguments, *args, **kwargs)
        except Exception as exc:
            if trace is not None:
                trace["calls"].append({**record, "seconds": round(time.time() - started, 3),
                                       "status": "exception", "error_type": type(exc).__name__})
            raise
        if trace is not None:
            text = _result_text(result)
            trace["calls"].append({
                **record, "seconds": round(time.time() - started, 3),
                "status": "tool_error" if getattr(result, "isError", False) else "ok",
                "result_chars": len(text),
                "result_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "result_preview": text[:_TRACE_PREVIEW_CHARS],
            })
        return result

    connector.call_tool = call_tool
    connector._cskg_trace_wrapped = True


async def run_mcp_agent(question: str) -> str:
    """
    Runs the MCPAgent with the given question and returns the result.

    Uses agent.stream()/_consume_and_return() instead of the simpler
    agent.run() specifically to recover `tools_used_names` -- run() calls
    the exact same internal machinery but discards that information,
    returning only the final text. Without it there was no way to tell
    "the agent actually queried the knowledge graph and found nothing" apart
    from "the agent never called a single tool and just answered from its
    own reasoning" -- both can produce fluent, plausible-sounding prose (see
    NO_DATA_MARKER's own docstring for a real example of the latter slipping
    through undetected). This is a structural ground-truth signal instead of
    inferring intent from natural language.
    """
    await ensure_cskg_ready()  # raises CskgDependencyError with the failing stage
    try:
        client = get_mcp_client()  # imports mcp_use as a side effect -- see module docstring
        from mcp_use import MCPAgent
        allowed = _allowed_tools(question)
        disallowed_tools = []
        agent_steps = MCP_MAX_STEPS
        if not client.get_all_active_sessions():
            await client.create_all_sessions()
        for session in client.get_all_active_sessions().values():
            _instrument_connector(session.connector)
        if allowed:
            # A focused ATT&CK mapping query has a single purpose-built tool.
            # Discover its registered name first, then hide the other 42
            # options from the agent so it cannot spend its budget on broad
            # searches or text_to_sparql's separate OpenAI API call.
            tool_names = []
            for session in client.get_all_active_sessions().values():
                tool_names.extend(tool.name for tool in await session.connector.list_tools())
            missing = [name for name in allowed if name not in tool_names]
            if missing:
                raise CskgDependencyError(
                    f"Required MCP tool is unavailable: {', '.join(missing)}", stage="agent_tool_missing")
            disallowed_tools = _disallow_tools_except(tool_names, allowed)
            # ATT&CK: one lookup, one fallback, one final response. CVE: up to
            # two product/ID lookups and one score/reference follow-up.
            agent_steps = min(MCP_MAX_STEPS, 5 if allowed is _CVE_TOOLS else 3)
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=agent_steps,
            verbose=True,
            system_prompt=strict_system_prompt,
            disallowed_tools=disallowed_tools,
        )
        generator = agent.stream(question, track_execution=False)
        trace = _TRACE.get()
        if trace is not None:
            trace.update(allowed_tools=list(allowed) if allowed else None, max_steps=agent_steps)
        result, steps_taken, tools_used_names = await agent._consume_and_return(generator)
        if trace is not None:
            trace.update(steps=steps_taken, tools_used=list(tools_used_names or []))
        # mcp_use returns its current text when max_steps is reached. That
        # text can omit required parts of the answer (for example,
        # mitigations after several broad searches), so it must not be
        # accepted as a successful graph lookup.
        _ensure_step_budget_not_exhausted(steps_taken, agent_steps)
        if tools_used_names:
            logger.info(f"[mcp_rdf_agent] question={question!r} steps={steps_taken} tools_used={tools_used_names}")
        else:
            logger.warning(
                f"[mcp_rdf_agent] question={question!r} answered WITHOUT calling any tool "
                f"(steps={steps_taken}) -- this is not a real knowledge-graph lookup, whatever the "
                f"response text claims"
            )
            if not str(result).startswith(NO_DATA_MARKER):
                # Fluent prose with zero tool calls came from the model's
                # own memory, not from the knowledge graph. It must not
                # reach the synthesizer as if it were CSKG evidence.
                return (f"{NO_DATA_MARKER}: the knowledge-base agent produced an answer without calling "
                        f"any knowledge-graph tool, so that answer was discarded as ungrounded.")
        return result
    except CskgDependencyError:
        raise
    except Exception as e:
        # Previously returned as an ordinary answer string ("Error during
        # MCP agent execution: ..."), which made a broken dependency look
        # like a retrieved result. Raised instead, so the dispatcher
        # records it as a dependency failure with its type and message.
        logger.error(f"An error occurred while running the MCP Agent: {type(e).__name__}: {e}")
        raise CskgDependencyError(
            f"CSKG agent run failed: {type(e).__name__}: {e or 'no message'}", stage="agent_run") from e
