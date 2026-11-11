# src/graph/workflow.py
"""
Rewired around question_generation_agent.py: instead of a fixed pipeline
(guardrails -> vector -> review -> cypher -> review -> log_analysis ->
mcp_rdf -> synthesizer) with a hardcoded reflection/retry loop
(max_iterations=3) at each step, the flow is now:

    guardrails (relevance only)
        -> question_generation (several hypothesis-tagged questions)
        -> dispatch_retrieval (routes each question to cypher/vector/cskg
                                per its `source` tag, concurrently)
        -> review_evidence (hypothesis verdict per question, not a retry gate)
        -> synthesizer (reasons over the hypothesis summary as a whole)
        -> grounding_check (unchanged: one bounded retry on ungrounded citations)

Dissolved, and why:
- reflection_agent.py (vector_reflection_chain, reflection_chain): its job
  was reactive rephrase-and-retry on a single failed question.
  question_generation_agent.py now asks several well-targeted questions up
  front, so a single empty result is informative (rules out a hypothesis),
  not a failure needing recovery.
- log_analysis_agent.py: its "decide cskg_required" job is now decided
  per-question, upfront, by question_generation_agent's `source` tag rather
  than after cypher/vector results come back. Its "summarize findings" job
  is now the deterministic (non-LLM) _render_hypothesis_summary below, fed
  into the synthesizer directly -- one fewer LLM call, not just a moved one.
- routing_agent.py: already dead code (never wired into this graph even
  before this change).

review_vector_answer/review_cypher_answer collapse into one
review_evidence node (review_agent.py's review_chain now runs once per
generated question rather than once per source), since there's no longer a
per-source retry branch that needed two separate conditional edges.

Two later additions adapt this same fixed topology to the multi-turn
investigation flow (src/investigation/session.py) without changing the
graph's edges -- both are node-internal short-circuits, not new routing:
- `state['skip_guardrails']` (turn 1 of a case investigation only):
  guardrails_node skips its LLM call entirely -- the input is a fixed
  directive over a case a human already selected, not a question needing
  triage.
- `state['investigation_mode']` (guardrails_agent.py's per-turn routing
  decision, meaningful on follow-up turns): 'full_hypothesis' behaves
  exactly as described above; 'direct_question' asks
  question_generation_node for exactly one targeted question instead of
  3-5; 'answer_from_context' skips question_generation/dispatch_retrieval/
  review_evidence entirely and lets synthesizer answer straight from
  whatever question_results/log_cypher_context/log_vector_context/
  mcp_rdf_context the last turn that actually retrieved something left
  behind -- these are plain (non-reducer) state fields, so the checkpointer
  carries them forward unchanged across a turn that does no new retrieval,
  which is what makes this "cache" possible with no extra state or
  storage.

Config v3 (after the russellmitchell-cases-fused pilot; see
eval/output/russellmitchell-cases-fused/baseline_status.json) changes what
three nodes do, again without changing the graph's edges:
- dispatch_retrieval: when `state['case_scope']` is set, Cypher and vector
  retrieval are restricted to the case's member alerts BEFORE any row
  limit, and everything else comes back as a separate, labelled historical
  channel (src/retrieval/case_scope.py). A retrieval that fails is recorded
  as a failure with its error type, stage and timing
  (`state['dependency_failures']`, `state['source_status']`) instead of
  becoming an "Error retrieving..." string indistinguishable from evidence.
- synthesizer: emits structured `claims`; reports unavailable sources as
  unavailable rather than "Not applicable".
- grounding_check: validates each claim against the alerts it cites and
  against case membership (src/agents/grounding_check.py::validate_report),
  not just that cited entity strings occur somewhere in the context;
  unsupported conclusions lower confidence and priority deterministically.
  By default the synthesizer rates each claim itself in the call that writes
  the report (SELF-VALIDATION in its prompt); EVIDENCE_CLAIM_VERIFIER=llm opts
  in to a separate model call per claim (src/agents/claim_verifier.py).
  Neither is human ground truth.
"""
import asyncio
import logging
import os
import time
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeoutError
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState

from src.agents.guardrails_agent import guardrails_chain, _render_alert_context
from src.agents.question_generation_agent import generate_questions
from src.agents.review_agent import review_chain
from src.agents.synthesizer_agent import synthesis_chain, synthesis_chain_fast, SynthesizedReport, answer_format
from src.agents.vector_agent import query_vector_search
from src.agents.cypher_agent import query_cypher
from src.agents.mcp_rdf_agent import NO_DATA_MARKER as _MCP_NO_DATA_MARKER, run_mcp_agent, start_trace as _start_cskg_trace
from src.agents.grounding_check import (
    apply_claim_verdicts, apply_self_assessment, case_context_alerts, retry_note as _validation_retry_note,
    summarize_claims, validate_report, validation_adjustments,
)
from src.agents import claim_verifier
from src.retrieval import case_scope as cs

logger = logging.getLogger(__name__)  # graph assembly / module-level only

# One dedicated logger per agent/node, named so %(name)s in the log line
# (see src/utils/logging_config.py's format string) identifies which agent
# produced it at a glance -- "agent.synthesizer" vs. "agent.guardrails" --
# instead of every node sharing this module's own logger name and relying
# on an ad-hoc text tag inside the message to tell them apart. Lets you
# filter one agent's output alone, e.g. `grep 'agent.synthesizer'`.
guardrails_logger = logging.getLogger("agent.guardrails")
question_generation_logger = logging.getLogger("agent.question_generation")
dispatch_retrieval_logger = logging.getLogger("agent.dispatch_retrieval")
review_evidence_logger = logging.getLogger("agent.review_evidence")
synthesizer_logger = logging.getLogger("agent.synthesizer")
grounding_check_logger = logging.getLogger("agent.grounding_check")

# A single retrieval attempt (LLM call(s) + Neo4j query) has no timeout of
# its own by default. Every retrieval call is run under a hard wall-clock
# deadline; a timeout is treated as a normal retrieval failure (the
# except blocks below already degrade to an error string in that
# question's context) rather than blocking indefinitely -- same fix as the
# 37/11-minute MCP SPARQL hangs (socket.setdefaulttimeout in
# mcp-cskg-rdf/server.py).
NODE_CALL_TIMEOUT_SECONDS = int(os.environ.get("NODE_CALL_TIMEOUT_SECONDS", "60"))
CSKG_NODE_CALL_TIMEOUT_SECONDS = int(os.environ.get("CSKG_NODE_CALL_TIMEOUT_SECONDS", "120"))
CYPHER_NODE_CALL_TIMEOUT_SECONDS = int(os.environ.get("CYPHER_NODE_CALL_TIMEOUT_SECONDS", "180"))
CYPHER_MAX_CONCURRENCY = max(1, int(os.environ.get("CYPHER_MAX_CONCURRENCY", "2")))
_retrieval_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieval-timeout")

# How many past turns' retrieved evidence dispatch_retrieval_node keeps
# accumulated in state['question_results'] (each entry tagged with the
# turn it came from) before dropping the oldest. Without this, a
# full_hypothesis/direct_question turn wholesale REPLACED the previous
# turn's question_results/log_cypher_context/log_vector_context/
# mcp_rdf_context (none of AgentState's fields besides `messages` have a
# LangGraph reducer, so a node's return value overwrites, never merges) --
# meaning a later 'answer_from_context' turn could only ever see the most
# recent retrieval-performing turn's evidence, not anything found earlier
# in the conversation. Bounded rather than unbounded so a long
# conversation doesn't grow this (and therefore every downstream LLM
# call's prompt size) forever.
EVIDENCE_WINDOW_TURNS = int(os.environ.get("EVIDENCE_WINDOW_TURNS", "5"))


def _window_question_results(results: list, current_turn: int) -> list:
    """Keeps only entries from the most recent EVIDENCE_WINDOW_TURNS
    distinct turns (including the current one) -- oldest turns dropped
    first, in whole-turn units (never splits a single turn's questions
    across the boundary). Entries missing a 'turn' tag (shouldn't happen
    going forward, but defensive against any pre-existing checkpointed
    state from before this field existed) are treated as belonging to the
    current turn rather than crashing."""
    turns_present = sorted({r.get('turn', current_turn) for r in results})
    keep_turns = set(turns_present[-EVIDENCE_WINDOW_TURNS:])
    return [r for r in results if r.get('turn', current_turn) in keep_turns]


def _call_with_timeout(fn, *args, timeout: int = NODE_CALL_TIMEOUT_SECONDS, **kwargs):
    future = _retrieval_executor.submit(copy_context().run, fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except _FutureTimeoutError:
        raise TimeoutError(
            f"{getattr(fn, '__name__', fn)} did not complete within {timeout}s"
        )


def _render_conversation_context(messages: list) -> str:
    """Renders every prior turn (all but the current turn's own trailing
    human message, which is already passed separately as `question`) as
    alternating Analyst/Investigator lines, for the case-investigation
    chatroom (src/investigation/session.py). 'N/A' on a fresh thread's
    first turn -- the whole point of checkpointing state['messages'] via
    src.investigation.session's checkpointed_app is so this has real
    history to render on later turns, rather than every follow-up being
    answered as if it were a brand new, context-free question."""
    if not messages or len(messages) <= 1:
        return "N/A (first turn of this investigation)"
    lines = []
    for m in messages[:-1]:
        role = "Analyst" if getattr(m, "type", None) == "human" else "Investigator"
        lines.append(f"{role}: {getattr(m, 'content', str(m))}")
    return "\n".join(lines)


def _render_prior_evidence(state: AgentState) -> str:
    """Compact summary of what's already been retrieved in this
    investigation thread (persists across checkpointed turns since
    question_results isn't a reducer -- see state.py), used by guardrails
    to judge whether a follow-up can be answered from it directly instead
    of triggering new retrieval (GuardrailsOutput.investigation_mode).
    'N/A' when nothing has been retrieved yet (turn 1, or src/run.py/
    eval/'s single-shot paths, which never populate question_results)."""
    summary = _render_hypothesis_summary(state.get('question_results') or [])
    return summary if summary != "Not applicable for this query." else "N/A (no evidence gathered yet)"


def _current_turn_number(messages: list) -> int:
    """1-indexed count of human messages so far, used to give each turn's
    answer AI message a stable id (see guardrails_node/synthesize_node) --
    `add_messages` replaces an existing message sharing an id instead of
    appending a duplicate, so a bounded grounding-check retry (synthesize_node
    running twice in one turn) doesn't leave two answers in the transcript."""
    return sum(1 for m in (messages or []) if getattr(m, "type", None) == "human") or 1


# --- Retrieval outcome bookkeeping ---
_SOURCES = ("cypher", "vector", "cskg")


def _describe_failure(source: str, exc: BaseException, attempts: int, seconds: float) -> dict:
    """A retrieval failure as data. `str(asyncio.TimeoutError())` is the
    empty string, which is how the pilot's CSKG outage ended up logged as
    "failed after 2 attempt(s): " with nothing after the colon."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) and not str(exc):
        limit = {'cskg': CSKG_NODE_CALL_TIMEOUT_SECONDS,
                 'cypher': CYPHER_NODE_CALL_TIMEOUT_SECONDS}.get(source, NODE_CALL_TIMEOUT_SECONDS)
        message = f"no response within {limit}s"
    else:
        message = str(exc) or "no message"
    return {"source": source, "error_type": type(exc).__name__, "message": message,
            "stage": getattr(exc, "stage", None), "attempts": attempts, "seconds": round(seconds, 1)}


def _retrieval_failed(result: dict) -> bool:
    return (result.get("retrieval") or {}).get("status") == "failed"


def _query_error(result: dict) -> bool:
    """The source was reachable, but the agent's own generated query was
    rejected (Cypher syntax/semantics, schema, placeholder anchor) on every
    attempt. That is a result of the pipeline under test -- it is recorded
    in the turn and counted -- not a dependency outage that invalidates the
    turn. It is never negative evidence either."""
    return (result.get("retrieval") or {}).get("status") == "query_error"


def _is_generation_fault(source: str, exc: BaseException) -> bool:
    if source != "cypher":
        return False
    from src.retrieval import scoped_fetch
    return isinstance(exc, ValueError) or scoped_fetch.is_cypher_syntax_error(exc)


def _has_case_evidence(result: dict) -> bool:
    return bool((result.get("evidence") or {}).get("case_alert_ids") or (result.get("evidence") or {}).get("case_rows"))


def _source_status(results: list) -> dict:
    """Per source: not_queried | ok | partial_failure | failed, with the
    failure messages. Computed from the same windowed list everything
    else is derived from, so a later answer_from_context turn still knows
    a source was down when its evidence was gathered."""
    status = {}
    for source in _SOURCES:
        mine = [r for r in results if r.get("source") == source]
        failed = [r for r in mine if _retrieval_failed(r)]
        query_errors = [r for r in mine if _query_error(r)]
        if not mine:
            state = "not_queried"
        elif not failed:
            state = "ok"
        else:
            state = "failed" if len(failed) == len(mine) else "partial_failure"
        status[source] = {"status": state, "questions": len(mine), "failed": len(failed),
                          "errors": sorted({f"{r['retrieval']['error_type']}: {r['retrieval']['message']}" for r in failed}),
                          "query_errors": len(query_errors),
                          "query_error_messages": sorted({f"{r['retrieval']['error_type']}: {r['retrieval']['message'][:300]}"
                                                          for r in query_errors})}
    return status


def _build_evidence_index(results: list, case_scope: dict | None, case_context: str | None) -> dict:
    """{alert_id: {scope, text, timestamp, host, time_relation, sources}}
    for every alert whose content the model was shown: scoped retrieval
    rows plus the member alerts quoted in the case fact sheet. This is
    what citations are validated against."""
    if not case_scope:
        return {}
    index: dict = {}
    for aid, line in case_context_alerts(case_context, case_scope["alert_ids"]).items():
        index[aid] = {"scope": cs.SCOPE_CASE, "text": line, "timestamp": None, "host": None,
                      "time_relation": "member", "sources": ["case_context"]}
    for r in results:
        evidence = r.get("evidence") or {}
        for aid, prov in (evidence.get("provenance") or {}).items():
            entry = index.setdefault(aid, {"scope": prov.get("scope"), "text": "", "timestamp": None, "host": None,
                                           "time_relation": prov.get("time_relation"), "sources": []})
            entry["timestamp"] = entry["timestamp"] or prov.get("timestamp")
            entry["host"] = entry["host"] or prov.get("host")
            text = (evidence.get("alert_text") or {}).get(aid) or ""
            if text and text not in entry["text"] and len(entry["text"]) < 2 * cs.EVIDENCE_TEXT_CHARS:
                entry["text"] = (entry["text"] + "\n" + text).strip()
            if r.get("source") not in entry["sources"]:
                entry["sources"].append(r.get("source"))
    return index


def _render_failures(failures: list) -> str:
    if not failures:
        return "None."
    return "\n".join(
        f"- {f['source']} (turn {f.get('turn', '?')}): {f['error_type']}: {f['message']} "
        f"[question: {f.get('question', '?')!r}; {f['attempts']} attempt(s), {f['seconds']}s]" for f in failures)


# --- Node Definition: Guardrails ---
def guardrails_node(state: AgentState):
    """Relevance check + investigation-mode routing in one call. Used to
    also decide escalation ("should this auto-triggered alert be
    investigated at all") -- removed once ingestion/investigation split
    into two processes (see guardrails_agent.py's module docstring):
    nothing auto-triggers the pipeline anymore, so there was nothing left
    to triage. An irrelevant question still ends the graph here, before
    the expensive downstream pipeline (question generation, three-way
    retrieval, review, synthesis) runs at all."""
    messages = state.get('messages') or []
    turn_no = _current_turn_number(messages)

    if state.get('skip_guardrails'):
        # Turn 1 of a case investigation: the input is a fixed, hardcoded
        # directive over a case a human already selected (see
        # src/investigation/session.py::start_investigation), not
        # analyst-authored text that could plausibly be off-topic -- and
        # there's no prior evidence yet either, so this is always a fresh
        # full-hypothesis investigation. Skips the LLM call entirely.
        guardrails_logger.info("SKIPPED (initial case investigation) -> relevant, full_hypothesis")
        return {"is_relevant": True, "investigation_mode": "full_hypothesis"}

    question = state['question']
    alert_context = _render_alert_context(state.get('alert'))
    case_context = (state.get('case_context') or "N/A") + (state.get('related_findings_context') or "")
    conversation_context = _render_conversation_context(messages)
    prior_evidence_summary = _render_prior_evidence(state)
    guardrails_logger.info(f"invoked -- question={question!r}")
    result = guardrails_chain.invoke({
        "question": question,
        "alert_context": alert_context,
        "case_context": case_context,
        "conversation_context": conversation_context,
        "prior_evidence_summary": prior_evidence_summary,
    })
    guardrails_logger.info(
        f"output: decision={result.decision} investigation_mode={result.investigation_mode} "
        f"reasoning={result.reasoning!r}"
    )

    if result.decision == "irrelevant":
        answer = "Sorry, I can only answer questions related to log analysis and cybersecurity knowledge topic"
        return {
            "is_relevant": False,
            "investigation_mode": result.investigation_mode,
            "answer": answer,
            "messages": [AIMessage(content=answer, id=f"turn-{turn_no}-answer")],
        }

    return {"is_relevant": True, "investigation_mode": result.investigation_mode}


# --- Node Definition: Question Generation ---
# The ID tool returns one row per technique, so this stays within the
# server's row cap; sorted order keeps the selection deterministic.
MITIGATION_LOOKUP_MAX_IDS = 10


def _guaranteed_mitigation_question(state: AgentState) -> dict | None:
    """Deterministic (non-LLM) mitigation-lookup question, appended to
    whatever question_generation produced this turn -- guarantees
    mitigation_suggestions can be grounded in real MITRE ATT&CK mitigation
    data (get_mitigations_for_technique, mcp-cskg-rdf/server.py) instead of
    being left up to chance whether the LLM's free hypothesis set happened
    to include a mitigation-shaped question (it typically hasn't --
    mitigation_suggestions has so far always been free-generated advice,
    never an actual CSKG lookup). None when no MITRE technique is known
    yet (state['known_mitre_techniques'], falling back to the single
    alert's own native tag for src/run.py/eval/) -- nothing to look up
    mitigations for."""
    techniques = state.get('known_mitre_techniques') or (state.get('alert') or {}).get('rule_mitre_id') or []
    if not techniques:
        return None
    ids = ", ".join(sorted(set(techniques))[:MITIGATION_LOOKUP_MAX_IDS])
    return {
        "hypothesis": "Standard mitigations for the identified technique(s)",
        # Only their IDs are ever known at this point (not names) -- the
        # question says so explicitly and names the exact tool sequence,
        # since get_mitigations_for_technique matches on name text, not ID
        # (a bare ID silently returns nothing -- see mcp_rdf_agent.py's
        # system prompt for the general version of this pointer; spelling
        # it out here too, per-question, is extra reinforcement specifically
        # because this exact question caused that failure once already).
        "question": (
            f"What are the recommended mitigations for MITRE ATT&CK technique ID(s) {ids}? "
            f"Use get_mitigations_for_technique_ids with exactly these IDs."
        ),
        "source": "cskg",
        "rationale": "Guaranteed lookup (not LLM-chosen) so mitigation_suggestions is grounded in "
        "documented MITRE mitigation data instead of free-generated advice.",
    }


def question_generation_node(state: AgentState):
    """Generates several hypothesis-tagged investigative questions from the
    alert/question, each pre-tagged with which retrieval source answers it."""
    mode = state.get('investigation_mode') or "full_hypothesis"
    if mode == "answer_from_context":
        # Guardrails decided the evidence already gathered in a prior turn
        # answers this follow-up -- skip generating (and therefore
        # retrieving/reviewing) anything new. question_results/
        # log_cypher_context/log_vector_context/mcp_rdf_context are left
        # untouched (returning {} here, not overwriting them), so they
        # still hold whatever the last turn that actually retrieved
        # something put there -- that's the "cache" synthesize_node reads.
        question_generation_logger.info("SKIPPED (mode=answer_from_context) -- reusing prior evidence")
        return {}

    alert_summary = state['original_question']
    case_context = (state.get('case_context') or "N/A") + (state.get('related_findings_context') or "")
    conversation_context = _render_conversation_context(state.get('messages') or [])
    question_generation_logger.info(f"invoked (mode={mode}, tier={'full' if mode in ('full_hypothesis', 'single_interpretation') else 'fast'})")
    result = generate_questions(alert_summary, case_context=case_context, conversation_context=conversation_context, mode=mode)
    questions = [q.model_dump() for q in result.questions]
    question_generation_logger.info(
        f"output: {len(questions)} question(s) across "
        f"{len(set(q['hypothesis'] for q in questions))} hypothesis/hypotheses"
    )
    for q in questions:
        question_generation_logger.info(
            f"  - hypothesis={q['hypothesis']!r} question={q['question']!r} source={q['source']}"
        )

    mitigation_q = _guaranteed_mitigation_question(state)
    if mitigation_q and mode != "single_interpretation":
        questions.append(mitigation_q)
        question_generation_logger.info(f"  - GUARANTEED: question={mitigation_q['question']!r} source=cskg")

    return {"generated_questions": questions}


# --- Node Definition: Dispatch Retrieval ---
async def dispatch_retrieval_node(state: AgentState):
    """Routes each generated question to cypher/vector/cskg per its `source`
    tag (a fixed rule, decided by question_generation_agent's prompt -- not
    a computed EIG/cost estimate), running all questions concurrently."""
    if state.get('investigation_mode') == "answer_from_context":
        # question_generation_node already skipped and returned {} this
        # turn, but `generated_questions` is a non-reducer field -- it
        # would still hold the *previous* turn's questions, not an empty
        # list, if we didn't check the mode explicitly here too. Skipping
        # explicitly (rather than trusting "no questions") is what
        # actually prevents re-dispatching stale questions and keeps
        # log_cypher_context/log_vector_context/mcp_rdf_context/
        # question_results as the cached evidence from whichever turn last
        # populated them.
        dispatch_retrieval_logger.info("SKIPPED (mode=answer_from_context) -- reusing cached evidence")
        return {}

    questions = state.get('generated_questions') or []
    if not questions:
        dispatch_retrieval_logger.info("no questions generated this turn -- nothing to dispatch")
        return {}
    query_timestamp = state.get('query_timestamp')
    case_scope = state.get('case_scope')
    turn_no = _current_turn_number(state.get('messages') or [])
    dispatch_retrieval_logger.info(
        f"invoked -- dispatching {len(questions)} question(s) concurrently "
        f"({'case-scoped to ' + case_scope['case_id'] if case_scope else 'unscoped'})")

    async def _retrieve_once(source: str, q_text: str):
        """One retrieval attempt for a single (source, question) pair.
        Raises on failure -- run_one below decides whether to retry."""
        if source == "vector":
            return await asyncio.to_thread(_call_with_timeout, query_vector_search, q_text, query_timestamp, case_scope)
        elif source == "cypher":
            async with cypher_slots:
                cypher_result = await asyncio.to_thread(
                    _call_with_timeout, query_cypher, q_text, query_timestamp, case_scope,
                    timeout=CYPHER_NODE_CALL_TIMEOUT_SECONDS)
            return cypher_result.get("context", [])
        elif source == "cskg":
            # The trace collects every tool call of this attempt; it is
            # attached to the question result by run_one.
            cskg_traces.setdefault(q_text, []).append(_start_cskg_trace())
            return await asyncio.wait_for(run_mcp_agent(q_text), timeout=CSKG_NODE_CALL_TIMEOUT_SECONDS)
        else:
            return f"Unknown source tag '{source}' for this question."

    cskg_traces: dict = {}
    _MAX_RETRIEVAL_ATTEMPTS = 2  # 1 bounded retry, same bound as grounding_check's synthesis retry elsewhere in this graph
    cypher_slots = asyncio.Semaphore(CYPHER_MAX_CONCURRENCY)

    async def run_one(gq: dict) -> dict:
        result = await _run_one(gq)
        if gq.get("source") == "cskg":
            # One entry per attempt (a retried question has two).
            result["cskg_trace"] = cskg_traces.get(gq.get("question"), [])
        return result

    async def _run_one(gq: dict) -> dict:
        """Retries only an actual retrieval *failure* (an exception --
        timeout, malformed query, missing index, etc.), never a
        legitimately empty/negative result: query_cypher/
        query_vector_search/run_mcp_agent returning "nothing found" is a
        normal return value, not an exception, so it flows straight
        through untouched -- an empty or negative result is itself
        evidence (rules out a hypothesis), not a failure to recover from,
        per question_generation_agent.py's module docstring. Only a
        genuine exception -- which conflates two different things a
        naive read of "no results" can't tell apart -- gets a second
        attempt, on the chance it was transient (a network blip, a
        momentary timeout) rather than a persistent bug that'll just fail
        the same way again."""
        q_text = gq["question"]
        source = gq["source"]
        context, failure, started = None, None, time.monotonic()
        max_attempts = 1 if source == "cskg" else _MAX_RETRIEVAL_ATTEMPTS
        for attempt in range(1, max_attempts + 1):
            try:
                context = await _retrieve_once(source, q_text)
                failure = None
                break
            except Exception as e:
                failure = _describe_failure(source, e, attempt, time.monotonic() - started)
                if _is_generation_fault(source, e):
                    # query_cypher already regenerated with the error shown;
                    # rerunning the same prompt would only repeat the fault.
                    failure["generation_fault"] = True
                    dispatch_retrieval_logger.warning(
                        f"({source}) {q_text!r} QUERY ERROR (agent-generated query invalid; recorded, "
                        f"not an outage): {failure['error_type']}: {failure['message'][:300]}")
                    break
                # A timed-out worker thread may still be running. Retrying it
                # here can overlap the same expensive request and amplify load.
                retryable = not isinstance(e, TimeoutError)
                if source == "cypher":
                    # A lost routing connection is worth one retry. Do not
                    # rerun a slow/failed database query and multiply load.
                    from neo4j.exceptions import ServiceUnavailable
                    retryable = isinstance(e, ServiceUnavailable)
                if attempt < max_attempts and retryable:
                    dispatch_retrieval_logger.warning(
                        f"({source}) {q_text!r} failed on attempt {attempt}, retrying once: "
                        f"{failure['error_type']}: {failure['message']}"
                    )
                    continue
                dispatch_retrieval_logger.error(
                    f"({source}) {q_text!r} DEPENDENCY FAILURE after {attempt} attempt(s) in {failure['seconds']}s: "
                    f"{failure['error_type']}: {failure['message']}"
                    + (f" (stage: {failure['stage']})" if failure['stage'] else ""))
        if failure and failure.get("generation_fault"):
            context = ("QUERY ERROR -- the query generated for this question was invalid on every attempt "
                       f"({failure['error_type']}: {failure['message'][:300]}). The question was NOT answered; "
                       "this is NOT a negative result.")
            return {**gq, "context": context, "retrieval": {**failure, "status": "query_error"}, "evidence": None}
        if failure:
            # Kept as a context string too (review/synthesis prompts read
            # it), but the structured `retrieval` record is what marks
            # this as a failure -- never as an empty or negative result.
            context = (f"RETRIEVAL FAILED -- the {source} source could not be queried "
                       f"({failure['error_type']}: {failure['message']}). This is NOT a negative result.")
            return {**gq, "context": context, "retrieval": {"status": "failed", **failure}, "evidence": None}
        evidence = None
        if isinstance(context, dict) and context.get("scoped"):
            evidence = {
                "scope_enforcement": context["scope_enforcement"],
                "case_alert_ids": context["case_evidence"]["alert_ids"],
                "historical_alert_ids": context["historical_evidence"]["alert_ids"],
                "case_rows": len(context["case_evidence"]["rows"]),
                "case_rows_total": context["case_evidence"]["total_rows"],
                "historical_rows": len(context["historical_evidence"]["rows"]),
                "historical_rows_total": context["historical_evidence"]["total_rows"],
                "provenance": context["provenance"], "alert_text": context["alert_text"],
                "queries": context.get("queries"), "notes": context.get("notes"),
            }
            dispatch_retrieval_logger.info(
                f"scope: ({source}) {q_text!r} -> {evidence['case_rows']}/{evidence['case_rows_total']} case-local, "
                f"{evidence['historical_rows']}/{evidence['historical_rows_total']} historical row(s) "
                f"[{evidence['scope_enforcement']}]" + (f" notes={evidence['notes']}" if evidence['notes'] else ""))
            context = context["rendered"]
        dispatch_retrieval_logger.info(f"output: ({source}) {q_text!r} -> {str(context)[:300]!r}")
        return {**gq, "context": context, "evidence": evidence,
                "retrieval": {"status": "ok", "attempts": attempt, "seconds": round(time.monotonic() - started, 1)}}

    results = list(await asyncio.gather(*(run_one(gq) for gq in questions)))
    for r in results:
        r["turn"] = turn_no  # tag before accumulating -- _window_question_results/review_evidence_node key on this

    # Accumulate onto prior turns' evidence rather than replacing it (see
    # EVIDENCE_WINDOW_TURNS' docstring above for why this matters), then
    # window. log_cypher_context/log_vector_context/mcp_rdf_context/
    # generated_question_for_rdf are DERIVED fresh from the windowed list
    # every time, rather than separately accumulated in parallel -- keeps
    # question_results as the single source of truth so these can't drift
    # out of sync with it.
    accumulated = (state.get('question_results') or []) + results
    windowed = _window_question_results(accumulated, turn_no)

    # Failed retrievals are excluded from the evidence contexts below --
    # they are reported through dependency_failures/source_status instead,
    # so an outage can neither count as a source "with data" nor be read
    # as "nothing found".
    usable = [r for r in windowed if not _retrieval_failed(r) and not _query_error(r)]

    def _has_rows(r):  # scoped results with no rows in either channel add nothing to the context
        e = r.get("evidence")
        return bool(r["context"]) and (e is None or e["case_rows"] or e["historical_rows"])

    cypher_ctx = [r["context"] for r in usable if r["source"] == "cypher" and _has_rows(r)]
    vector_ctx = "\n".join(str(r["context"]) for r in usable if r["source"] == "vector" and _has_rows(r))
    cskg_ctx = "\n".join(str(r["context"]) for r in usable if r["source"] == "cskg" and r["context"])
    cskg_questions = "; ".join(r["question"] for r in windowed if r["source"] == "cskg")
    failures = [{**r["retrieval"], "question": r["question"], "turn": r.get("turn")}
                for r in windowed if _retrieval_failed(r)]
    source_status = _source_status(windowed)
    if failures:
        dispatch_retrieval_logger.error(
            "DEPENDENCY FAILURES this investigation: "
            + "; ".join(f"{k}={v['status']} ({v['failed']}/{v['questions']})" for k, v in source_status.items()
                        if v["failed"]))

    dispatch_retrieval_logger.info(
        f"accumulated evidence: {len(windowed)} question(s) across "
        f"{len(set(r['turn'] for r in windowed))} turn(s) (window={EVIDENCE_WINDOW_TURNS} turns)"
    )

    return {
        "question_results": windowed,
        "log_cypher_context": cypher_ctx or None,
        "log_vector_context": vector_ctx or None,
        "mcp_rdf_context": cskg_ctx or None,
        "generated_question_for_rdf": cskg_questions or "Not applicable for this query.",
        "dependency_failures": failures,
        "source_status": source_status,
        "evidence_index": _build_evidence_index(windowed, case_scope, state.get('case_context')),
    }


# --- Node Definition: Review Evidence ---
def review_evidence_node(state: AgentState):
    """Reviews each question's retrieved evidence against the hypothesis it
    was meant to discriminate (review_agent.py). No retry branch: results
    (including ruled-out hypotheses) flow straight to the synthesizer."""
    if state.get('investigation_mode') == "answer_from_context":
        # dispatch_retrieval_node already skipped this turn -- nothing new
        # to review. question_results still holds whichever prior turn's
        # (already-reviewed) results, reused as-is by the synthesizer.
        review_evidence_logger.info("SKIPPED (mode=answer_from_context) -- nothing new to review")
        return {}
    results = state.get('question_results') or []
    # question_results now accumulates across turns (dispatch_retrieval_node,
    # EVIDENCE_WINDOW_TURNS) -- most entries here on turn 2+ already have a
    # "review" from whenever they were first retrieved. Re-reviewing them
    # again every turn would be wasted LLM calls at best and risk a verdict
    # silently flipping between turns at worst; only entries missing a
    # review (this turn's freshly-dispatched questions) get one.
    already_reviewed = sum(1 for r in results if r.get("review"))
    review_evidence_logger.info(
        f"invoked -- {len(results) - already_reviewed} new question(s) to review, "
        f"{already_reviewed} already reviewed in a prior turn reused as-is"
    )
    reviewed = []
    for r in results:
        if r.get("review"):
            reviewed.append(r)
            continue
        if _retrieval_failed(r):
            # Nothing was retrieved, so there is nothing to judge -- and
            # asking a model to weigh an error message risks it "ruling
            # out" a hypothesis that was never tested.
            failure = r["retrieval"]
            reviewed.append({**r, "review": {
                "sufficient": False, "hypothesis_verdict": "inconclusive",
                "reasoning": f"Not tested: the {r.get('source')} source could not be queried "
                             f"({failure['error_type']}: {failure['message']})."}})
            review_evidence_logger.warning(f"output: {r.get('question')!r} -> NOT TESTED (retrieval failed)")
            continue
        if _query_error(r):
            failure = r["retrieval"]
            reviewed.append({**r, "review": {
                "sufficient": False, "hypothesis_verdict": "inconclusive",
                "reasoning": f"Not tested: the query generated for this question was invalid "
                             f"({failure['error_type']}: {failure['message'][:200]})."}})
            review_evidence_logger.warning(f"output: {r.get('question')!r} -> NOT TESTED (invalid generated query)")
            continue
        review = review_chain.invoke({
            "hypothesis": r.get("hypothesis", ""),
            "question": r.get("question", ""),
            "context": str(r.get("context", "")),
        })
        review_evidence_logger.info(
            f"output: {r.get('question')!r} -> {review.hypothesis_verdict} ({review.reasoning!r})"
        )
        reviewed.append({**r, "review": review.model_dump()})
    return {"question_results": reviewed}


def _render_hypothesis_summary(question_results: list) -> str:
    """Deterministic (non-LLM) rollup of generated questions + review
    verdicts, replacing log_analysis_agent.py's old free-text summary --
    see the module docstring above for why this is one fewer LLM call, not
    a moved one. question_results can now span multiple turns
    (EVIDENCE_WINDOW_TURNS), so each line is tagged with which turn it's
    from -- without that, older and newer findings would be indistinguishable
    once accumulated together."""
    if not question_results:
        return "Not applicable for this query."
    lines = []
    for r in question_results:
        review = r.get("review", {})
        turn_tag = f"Turn {r['turn']} | " if r.get("turn") is not None else ""
        lines.append(
            f"- {turn_tag}Hypothesis: {r.get('hypothesis', '?')} | Question: {r.get('question', '?')} | "
            f"Source: {r.get('source', '?')} | Verdict: {review.get('hypothesis_verdict', '?')} "
            f"({review.get('reasoning', '')})"
        )
    return "\n".join(lines)


_NOT_APPLICABLE = "Not applicable for this query."
_NO_MCP_DATA = "No data was provided from this source."
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_MAX_CONFIDENCE_BY_RICHNESS = {0: "low", 1: "medium", 2: "high", 3: "high"}

# run_mcp_agent() (src/agents/mcp_rdf_agent.py) wraps a full LLM agent, not
# a raw tool passthrough like query_cypher/query_vector_search -- its
# output is always natural-language prose, whether it found something or
# not, so an exact-string check like _NO_MCP_DATA (which only ever matches
# the *default* set when no cskg question was asked at all -- see below)
# can't tell those two cases apart. NO_DATA_MARKER (mcp_rdf_agent.py) is
# the reliable signal for that; this phrase list is a defense-in-depth
# fallback for whenever the model doesn't prefix its answer with the
# marker exactly as instructed. Deliberately conservative (only common,
# unambiguous "found nothing" phrasings) -- a false negative here just
# means richness undercounts by one, which only ever pushes confidence
# *down*, never invents false certainty; a false positive would do the
# opposite, which is the actual failure mode being fixed.
_MCP_NO_DATA_PHRASES = (
    "couldn't find", "could not find", "no relevant information",
    "no known", "does not match", "doesn't match", "unable to find",
    "i don't have access", "i currently don't have access",
    "no results found", "no data found",
    # Added after a real miss: a question asking to correlate a specific IP
    # against threat intel (something this KB structurally has no data for
    # at all -- see mcp_rdf_agent.py's scope note) got answered with a
    # capability disclaimer + an offer to help with a *different*,
    # reformulated question instead -- not caught by NO_DATA_MARKER (the
    # model didn't use it) or any phrase above.
    "unable to directly analyze", "cannot directly analyze",
    "unable to directly access", "outside the scope of this knowledge base",
    "not something this knowledge base",
)


def _mcp_context_has_data(mcp_str: str) -> bool:
    if not mcp_str or mcp_str in (_NO_MCP_DATA, _NOT_APPLICABLE):
        return False
    if mcp_str.startswith("Error") or mcp_str.startswith(_MCP_NO_DATA_MARKER):
        return False
    lowered = mcp_str.lower()
    return not any(phrase in lowered for phrase in _MCP_NO_DATA_PHRASES)


def _context_richness(log_cypher: str, log_vector: str, mcp_rdf_context) -> int:
    """Counts how many of the 3 possible evidence sources actually returned
    substantive content, vs. the LLM's own unconstrained self-assessment
    (which eval showed always says 'high' regardless of whether
    the answer was actually correct -- confidence wasn't discriminating at
    all). This is the deterministic signal `confidence` gets capped against
    below, so stated confidence tracks actual evidence availability instead
    of vibes."""
    sources_with_data = 0
    if log_cypher and log_cypher not in (_NOT_APPLICABLE, "[]", "None"):
        sources_with_data += 1
    if log_vector and log_vector != _NOT_APPLICABLE:
        sources_with_data += 1
    if _mcp_context_has_data(str(mcp_rdf_context) if mcp_rdf_context else ""):
        sources_with_data += 1
    return sources_with_data


def _cap_confidence(report, richness: int):
    """Caps report.confidence at the max level `richness` evidence sources
    can support -- 0 sources -> at most 'low', 1 -> at most 'medium', 2+ ->
    'high' allowed. Only ever lowers confidence, never raises it (the LLM's
    own judgment can still say something is less confident than the cap)."""
    allowed_max = _MAX_CONFIDENCE_BY_RICHNESS[richness]
    if _CONFIDENCE_RANK[report.confidence] > _CONFIDENCE_RANK[allowed_max]:
        synthesizer_logger.info(
            f"capping confidence {report.confidence} -> {allowed_max} "
            f"({richness} of 3 evidence sources had substantive content)"
        )
        return report.model_copy(update={"confidence": allowed_max})
    return report


# --- Node Definition: Synthesizer ---
def _render_cypher_context(ctx) -> str:
    """Scoped results are already rendered text (one block per question);
    legacy unscoped results are row lists and keep their old str() form."""
    if not ctx:
        return _NOT_APPLICABLE
    if all(isinstance(c, str) for c in ctx):
        return "\n\n".join(ctx)
    return str(ctx)


def _cskg_has_data(state: AgentState) -> bool:
    """True if at least one knowledge-base question returned real data.
    Judged per question: joining the answers first (as mcp_rdf_context
    does for the prompt) let one "no relevant data" answer mask a second
    answer that did contain mitigations -- seen in the first live v3 run."""
    results = [r for r in (state.get('question_results') or []) if r.get("source") == "cskg"]
    if results:
        return any(not _retrieval_failed(r) and _mcp_context_has_data(str(r.get("context") or "")) for r in results)
    return _mcp_context_has_data(str(state.get('mcp_rdf_context') or ""))


def _annotate_report(report_dict: dict, state: AgentState) -> dict:
    """Deterministic facts about the evidence behind a report, written by
    the workflow rather than left to the model's own account of it."""
    status = state.get('source_status') or {}
    failures = state.get('dependency_failures') or []
    cskg = status.get("cskg") or {}
    cskg_has_data = _cskg_has_data(state)
    report_dict["source_status"] = status
    report_dict["dependency_failures"] = failures
    report_dict["mitigations_grounded_in_cskg"] = bool(report_dict.get("mitigation_suggestions")) and cskg_has_data
    if cskg.get("status") == "failed":
        # The pilot's reports said "Not applicable for this query." here
        # while every CSKG lookup had timed out.
        report_dict["cskg_context_summary"] = (
            "UNAVAILABLE: the cybersecurity knowledge base could not be queried "
            f"({'; '.join(cskg.get('errors') or ['unknown error'])}). Mitigation suggestions in this report "
            "are NOT backed by knowledge-base data.")
    return report_dict


def synthesize_node(state: AgentState):
    """Generates the final compiled report for the user (structured output,
    see src/agents/synthesizer_agent.py::SynthesizedReport)."""
    log_cypher = _render_cypher_context(state.get('log_cypher_context'))
    log_vector = str(state.get('log_vector_context')) if state.get('log_vector_context') else "Not applicable for this query."
    generated_q = str(state.get('generated_question_for_rdf', "Not applicable for this query."))
    hypothesis_summary = _render_hypothesis_summary(state.get('question_results') or [])
    messages = state.get('messages') or []
    turn_no = _current_turn_number(messages)
    case_context = (state.get('case_context') or "N/A") + (state.get('related_findings_context') or "")
    conversation_context = _render_conversation_context(messages)
    mode = state.get('investigation_mode') or "full_hypothesis"
    chain = synthesis_chain if mode in ("full_hypothesis", "single_interpretation") else synthesis_chain_fast
    failures = state.get('dependency_failures') or []
    source_status = state.get('source_status') or {}
    synthesizer_logger.info(f"invoked (mode={mode}, tier={'full' if mode in ('full_hypothesis', 'single_interpretation') else 'fast'})")

    if not state.get('mcp_rdf_context') and log_cypher == "Not applicable for this query." and log_vector == "Not applicable for this query.":
        if failures:
            # No evidence because sources FAILED is a different statement
            # from "nothing relevant exists" -- never report an outage as
            # a negative finding.
            analysis = ("No evidence could be retrieved: " + "; ".join(
                f"{f['source']} failed ({f['error_type']}: {f['message']})" for f in failures) + ".")
            answer = ("This could not be investigated: the evidence sources failed, so the absence of findings "
                      "below says nothing about the activity itself. " + analysis)
        else:
            analysis = "No data was retrieved from any source."
            answer = "Sorry, no relevant information could be found for this alert."
        synthesizer_logger.info(f"output: no evidence from any source -- returning fallback report ({analysis})")
        fallback = SynthesizedReport(
            original_question=state['original_question'],
            cypher_context_summary="Not applicable for this query.",
            vector_context_summary="Not applicable for this query.",
            generated_question_for_rdf="Not applicable for this query.",
            cskg_context_summary="Not applicable for this query.",
            critical_analysis=analysis,
            contextual_linkage="Not applicable for this query.",
            final_answer=answer,
            cited_entities=[],
            mitre_techniques=[],
            confidence="low",
            recommended_priority="monitor",
            mitigation_suggestions=[],
        )
        return {
            "answer": fallback.final_answer,
            # `fallback` tells grounding_check_node this report was written
            # by the workflow, not a model: it asserts nothing to validate.
            "synthesized_report": _annotate_report({**fallback.model_dump(), "fallback": True}, state),
            "messages": [AIMessage(content=fallback.final_answer, id=f"turn-{turn_no}-answer")],
        }

    # A previous grounding_check pass (see grounding_check_node /
    # decide_after_grounding below) may have rejected this report: cited
    # entities absent from context, citations outside the case, or claims
    # their own cited alerts do not support. Bounded to a single retry.
    previous = state.get('grounding_result') or {}
    is_retry = bool(previous) and not previous.get('valid', previous.get('grounded', True))
    retry_count = state.get('grounding_retry_count', 0) + (1 if is_retry else 0)
    retry_note = _validation_retry_note(previous) if is_retry else ""
    if is_retry and not retry_note:
        retry_note = (
            f"\n**IMPORTANT CORRECTION**: A previous draft cited entities not present in the "
            f"context above: {previous.get('ungrounded_entities', [])}. Do not repeat that mistake -- only cite "
            f"entities that literally appear in the context above."
        )

    cskg_status = (source_status.get("cskg") or {}).get("status")
    if state.get('mcp_rdf_context'):
        mcp_context = str(state['mcp_rdf_context'])
    elif cskg_status == "failed":
        mcp_context = ("UNAVAILABLE -- the knowledge base could not be queried (see Retrieval failures). "
                       "Do not present technique descriptions or mitigations as knowledge-base-backed.")
    else:
        mcp_context = _NO_MCP_DATA

    report = chain.invoke({
        "original_question": state['original_question'],
        "hypothesis_summary": hypothesis_summary,
        "log_cypher_context": log_cypher,
        "log_vector_context": log_vector,
        "generated_question_for_rdf": generated_q,
        "mcp_rdf_context": mcp_context,
        "retrieval_failures": _render_failures(failures),
        "retry_note": retry_note,
        "case_context": case_context,
        "conversation_context": conversation_context,
        "answer_format": answer_format(turn_no > 1),
    })

    richness = _context_richness(log_cypher, log_vector, state.get('mcp_rdf_context'))
    if state.get('case_scope'):
        # For a case, a source counts toward confidence only if it returned
        # evidence from the case itself; historical-only hits say nothing
        # about what happened here.
        results = [r for r in (state.get('question_results') or []) if not _retrieval_failed(r)]
        richness = (
            int(any(r["source"] == "cypher" and _has_case_evidence(r) for r in results))
            + int(any(r["source"] == "vector" and _has_case_evidence(r) for r in results))
            + int(_cskg_has_data(state))
        )
    raw_draft = report.model_dump()
    if state.get('enforcement_policy', 'complete') != 'diagnostic_only':
        report = _cap_confidence(report, richness)

    synthesizer_logger.info(
        f"output: final_answer={report.final_answer!r} confidence={report.confidence} "
        f"priority={report.recommended_priority} mitre_techniques={report.mitre_techniques} "
        f"cited_entities={report.cited_entities} claims={len(report.claims)} "
        f"mitigation_suggestions={report.mitigation_suggestions}"
    )

    # Grows the set _guaranteed_mitigation_question (question_generation_node)
    # draws on for later turns -- a technique newly confirmed THIS turn
    # gets a guaranteed mitigation lookup on the NEXT turn that needs new
    # retrieval, even if it wasn't already tagged on the case/alert.
    known_techniques = sorted(set(state.get('known_mitre_techniques') or []) | set(report.mitre_techniques))

    return {
        "answer": report.final_answer,
        "synthesized_report": _annotate_report(report.model_dump(), state),
        "raw_synthesis_draft": raw_draft,
        "evidence_richness": richness,
        "grounding_retry_count": retry_count,
        "known_mitre_techniques": known_techniques,
        # Same id every time this node runs within one turn (including a
        # bounded grounding-check retry) -- add_messages replaces the prior
        # message sharing an id instead of appending a duplicate, so the
        # transcript only ever shows this turn's final, corrected answer.
        "messages": [AIMessage(content=report.final_answer, id=f"turn-{turn_no}-answer")],
    }


# --- Node Definition: Grounding Check ---
_CLAIM_PROBLEMS = {
    "historical_cited_as_case": "cites an alert that is not part of this case",
    "unknown_alert": "cites an alert ID that matches no alert",
    "alert_not_in_evidence": "cites an alert whose content was never retrieved",
    "no_evidence": "cites no alert",
    "entity_not_in_cited_evidence": "names something the alerts it cites do not mention",
    "knowledge_base_unavailable": "relies on the knowledge base, which returned nothing",
    "partially_supported": "goes beyond what the alerts it cites show",
    "unsupported": "is not shown by the alerts it cites",
    "contradicted": "is contradicted by the alerts it cites",
    "unverified": "could not be verified",
    "not_found_with_citations": "says nothing was found yet cites alerts",
}


def _claim_problem(claim: dict) -> str:
    key = claim["structural_issues"][0].split(":", 1)[0] if claim["structural_issues"] else claim["support"]
    return _CLAIM_PROBLEMS.get(key, key.replace("_", " "))


def _validation_notice(result: dict, adjustments: dict) -> str:
    """Plain-language line appended to the answer when validation changed
    what the report may assert, so neither the analyst nor a later turn
    reading the transcript takes an unsupported conclusion as settled."""
    failed = [c for c in result.get("claims") or [] if not c["valid"]]
    parts = []
    if failed:
        parts.append(f"{len(failed)} of {len(result['claims'])} claim(s) in this answer did not pass validation: "
                     + "; ".join(f"\"{c['claim']}\" {_claim_problem(c)}" for c in failed[:5]))
    other = [v["detail"] for v in result.get("violations") or [] if v.get("blocking") and "claim_index" not in v]
    if other:
        parts.append("; ".join(other[:3]))
    for field, change in adjustments.items():
        parts.append(f"{field.replace('_', ' ')} lowered from {change['from']} to {change['to']} ({change['reason']})")
    return ("[Validation] " + ". ".join(p.rstrip(".") for p in parts) + ".") if parts else ""


def grounding_check_node(state: AgentState):
    """Validates the synthesizer's report. Entity presence as before; for
    case investigations also every claim against the alerts it cites and
    against case membership. Whether those alerts support each claim is
    taken from the synthesizer's own rating (no model call here); with
    EVIDENCE_CLAIM_VERIFIER=llm a separate model call judges it instead."""
    report = state.get('synthesized_report')
    if not report:
        grounding_check_logger.info("no report to check -- treating as grounded")
        return {"grounding_result": {"grounded": True, "valid": True, "checked_entities": [],
                                     "ungrounded_entities": [], "violations": [], "claims": []}}

    context_blobs = [
        str(state.get('log_cypher_context') or ''),
        str(state.get('log_vector_context') or ''),
        str(state.get('mcp_rdf_context') or ''),
        # case_context (src/investigation/narrative.py's static fact sheet:
        # hosts, IPs, users, MITRE tags, member-alert narratives) has been a
        # real input to synthesize_node's prompt since case-based
        # investigation was added, but was never added here -- so a
        # citation the synthesizer correctly grounded in the case's own
        # alert data (e.g. its source IP or host, stated directly in
        # case_context) was being flagged as a false-positive hallucination
        # and burning a retry for no reason. Deliberately NOT adding
        # conversation_context or hypothesis_summary here too: both contain
        # LLM-authored text (prior answers, review reasoning), and treating
        # those as grounding sources would let an entity that slipped
        # through once keep re-"grounding" itself in every later check --
        # case_context is safe specifically because it's built
        # deterministically from real ingested alert data, not from
        # another LLM's output.
        state.get('case_context') or '',
    ]
    if report.get('fallback'):
        grounding_check_logger.info("deterministic no-evidence report -- nothing to validate")
        return {"grounding_result": {"grounded": True, "valid": True, "final": True, "checked_entities": [],
                                     "ungrounded_entities": [], "violations": [], "claims": [],
                                     "claim_verifier": {"status": "not_applicable"}, "case_scoped": False,
                                     "summary": summarize_claims({})}}
    case_scope = state.get('case_scope')
    evidence_index = state.get('evidence_index') or _build_evidence_index(
        state.get('question_results') or [], case_scope, state.get('case_context'))
    result = validate_report(
        report, context_blobs=context_blobs, evidence_index=evidence_index, case_scope=case_scope,
        cskg_has_data=_cskg_has_data(state))

    verifier = {"status": "not_applicable"}
    if case_scope and result["claims"]:
        support_mode = claim_verifier.mode()
        if support_mode == "off":
            verifier = {"status": "disabled"}
        elif support_mode == "self":
            result = apply_self_assessment(result)
            verifier = {"status": "self_assessed",
                        "claims_judged": sum(c.get("support_source") == "self" for c in result["claims"]),
                        "claims_not_rated": sum(v["type"] == "not_self_assessed" for v in result["violations"])}
        else:
            try:
                verdicts = claim_verifier.verify_claims(result, evidence_index)
                result = apply_claim_verdicts(result, verdicts)
                verifier = {"status": "ok", "claims_judged": len(verdicts)}
            except Exception as e:
                # The verifier is a dependency too: its failure is recorded,
                # and claims stay 'unverified' rather than passing silently.
                verifier = {"status": "failed", "error_type": type(e).__name__, "message": str(e) or "no message"}
                result = apply_claim_verdicts(result, {})
                grounding_check_logger.error(f"claim verifier FAILED: {verifier['error_type']}: {verifier['message']}")
    result["claim_verifier"] = verifier
    result["summary"] = summarize_claims(result)

    if state.get('enforcement_policy', 'complete') == 'diagnostic_only':
        result.update(final=True, enforcement='diagnostic_only', adjustments={})
        return {"grounding_result": result}

    if result["valid"]:
        grounding_check_logger.info(
            f"output: valid=True grounded={result['grounded']} checked_entities={result['checked_entities']} "
            f"claims={result['summary']}")
    else:
        grounding_check_logger.warning(
            f"output: valid=False grounded={result['grounded']} ungrounded_entities={result['ungrounded_entities']} "
            f"violations={[v['detail'] for v in result['violations']]}")

    # Final for this turn unless decide_after_grounding will send it back
    # for the single retry. Once final, what the evidence does not support
    # is taken out of the report's confidence and priority deterministically.
    final = result["valid"] or state.get('grounding_retry_count', 0) >= 1
    result["final"] = final
    update = {"grounding_result": result}
    if final:
        adjustments = validation_adjustments(report, result)
        result["adjustments"] = adjustments
        notice = _validation_notice(result, adjustments)
        revised = {**report,
                   **{field: change["to"] for field, change in adjustments.items()},
                   "validation": {"valid": result["valid"], "summary": result["summary"], "adjustments": adjustments,
                                  "citations": result.get("citations"), "claim_verifier": verifier,
                                  "violations": [v["detail"] for v in result["violations"]]},
                   "validation_notice": notice}
        if (not result['valid'] or any(not c['valid'] for c in result['claims'])) and report.get('maliciousness') in ('malicious', 'benign'):
            revised['maliciousness'] = 'undetermined'
            revised['uncertainty'] = list(report.get('uncertainty') or []) + [
                'One or more claims failed validation; the conclusive case assessment is withheld.']
        if not result['valid'] or any(not c['valid'] for c in result['claims']):
            # A notice alone leaves the unsupported diagnosis in the headline.
            # Preserve the draft for audit, but publish only accepted statements.
            revised['unvalidated_draft'] = {key: report.get(key) for key in
                                            ('final_answer', 'critical_analysis', 'contextual_linkage')}
            statements = [f"- [{c['declared_scope']}] {c['claim']}" for c in result['claims']
                          if c['valid'] and c['support'] == 'supported']
            revised['final_answer'] = (
                'Evidence-supported statements:\n' + ('\n'.join(statements) or 'None passed claim validation.')
                + '\n\nAssessment: parts of the draft could not be validated; no conclusive diagnosis is presented.'
                + f"\nRecommended action: {revised.get('recommended_priority', 'monitor')}.")
            revised['critical_analysis'] = 'Unvalidated conclusions withheld; see validation details and saved draft.'
            revised['contextual_linkage'] = 'Only the accepted claims above may be used as established findings.'
        update["synthesized_report"] = revised
        if adjustments:
            grounding_check_logger.warning(f"validation adjustments applied: {adjustments}")
        if notice:
            turn_no = _current_turn_number(state.get('messages') or [])
            answer = f"{revised.get('final_answer')}\n\n{notice}"
            update["answer"] = answer
            update["messages"] = [AIMessage(content=answer, id=f"turn-{turn_no}-answer")]
    return update


# --- Perakitan Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("guardrails", guardrails_node)
workflow.add_node("question_generation", question_generation_node)
workflow.add_node("dispatch_retrieval", dispatch_retrieval_node)
workflow.add_node("review_evidence", review_evidence_node)
workflow.add_node("synthesizer", synthesize_node)
workflow.add_node("grounding_check", grounding_check_node)


# 1. Decision after Guardrails
def decide_after_guardrails(state: AgentState):
    if not state.get('is_relevant', False):
        guardrails_logger.info("routing: irrelevant -> ending execution")
        return END
    return "question_generation"


# 2. Decision after Grounding Check
def decide_after_grounding(state: AgentState):
    if state.get('enforcement_policy', 'complete') == 'diagnostic_only':
        return END
    result = state.get('grounding_result') or {}
    if result.get('valid', result.get('grounded', True)):
        grounding_check_logger.info("routing: report validated -> ending execution")
        return END
    if state.get('grounding_retry_count', 0) >= 1:
        grounding_check_logger.warning(
            "routing: still failing validation after retry -> surfacing with validation notice (bounded retry)")
        return END
    grounding_check_logger.warning("routing: report failed validation -> retrying synthesis once")
    return "synthesizer"


# --- Define Edges ---
workflow.set_entry_point("guardrails")

workflow.add_conditional_edges(
    "guardrails",
    decide_after_guardrails,
    {
        "question_generation": "question_generation",
        END: END
    }
)

workflow.add_edge("question_generation", "dispatch_retrieval")
workflow.add_edge("dispatch_retrieval", "review_evidence")
workflow.add_edge("review_evidence", "synthesizer")
workflow.add_edge("synthesizer", "grounding_check")

workflow.add_conditional_edges(
    "grounding_check",
    decide_after_grounding,
    {
        "synthesizer": "synthesizer",
        END: END
    }
)

# Compile graph
app = workflow.compile()
