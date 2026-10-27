# src/agents/cskg_health.py
"""
Independent health check for the CSKG (MITRE ATT&CK) MCP server.

"Independent" means it does not go through the LLM agent or mcp_use: it
reads the same browser_mcp.json, starts the same server process with the
official MCP client, and calls tools directly. A failure is therefore a
fact about the dependency, not about a prompt or a model.

Why this exists (pilot run russellmitchell-cases-fused, config v2): every
CSKG lookup timed out after 60 s, twice, with an EMPTY error message
(`str(asyncio.TimeoutError())` is ''), the report said "Not applicable for
this query" for the knowledge-base section, and the run was recorded as a
normal "fused" run. browser_mcp.json pointed the server at an interpreter
whose environment was offloaded by cloud file sync, so the process hung
on import. Nothing in the pipeline said so.

Each stage reports ok/failed with timing, the error class, the server's
own stderr where relevant, and a remediation hint:

    config          browser_mcp.json parses and names one server
    interpreter     the configured command exists, is local (not an
                    offloaded cloud placeholder), and starts
    imports         the server's dependencies import in that interpreter
    server_script   the server script exists and is local
    sparql_endpoint the configured SPARQL endpoint answers a trivial query
                    (checked from THIS process, bypassing the server)
    handshake       the server process completes MCP `initialize`
    tools           the tools the pipeline relies on are registered
    lookup          get_technique_by_id / get_mitigations_for_technique
                    return real MITRE data

Usage:
    python -m src.agents.cskg_health            # human-readable, exit 1 on failure
    python -m src.agents.cskg_health --json

This module deliberately imports nothing from src.config.settings so the
check also runs when Neo4j/OpenAI settings are broken.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "browser_mcp.json"

# Placeholders accepted in browser_mcp.json's command/args, so the file
# does not have to hard-code one machine's virtualenv path.
PLACEHOLDERS = {"${CURRENT_PYTHON}": lambda: sys.executable, "${PROJECT_ROOT}": lambda: str(PROJECT_ROOT)}
PYTHON_OVERRIDE_ENV = "CSKG_PYTHON"

SERVER_IMPORTS = ("mcp.server.fastmcp", "rdflib", "rdflib.plugins.stores.sparqlstore", "tiktoken", "openai", "dotenv")
# Imported by the server inside try/except, so their absence does not stop
# it -- but changes how it behaves. `truststore` makes the server verify
# HTTPS against the OS trust store; without it the public SEPSES endpoint
# fails certificate verification on some Python builds. It is listed in
# mcp-cskg-rdf/requirements.txt but was missing from the project's own
# pyproject.toml, so an environment built from uv.lock did not have it.
OPTIONAL_SERVER_IMPORTS = ("truststore",)
REQUIRED_TOOLS = ("get_technique_by_id", "get_mitigations_for_technique",
                  "get_technique_mitigations_by_keyword", "text_to_sparql", "health_check")
PROBE_TECHNIQUE_ID, PROBE_TECHNIQUE_NAME = "T1071", "Application Layer Protocol"

_SF_DATALESS = 0x40000000  # macOS: file content is not on disk (cloud placeholder)
_SCAN_LIMIT = 200_000


class CskgDependencyError(RuntimeError):
    """The CSKG could not be used. Carries the failing stage so the
    workflow can record a dependency failure instead of an empty string."""

    def __init__(self, message: str, stage: Optional[str] = None, report: Optional[dict] = None):
        super().__init__(message)
        self.stage, self.report = stage, report


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def _substitute(value: str) -> str:
    for token, resolver in PLACEHOLDERS.items():
        if token in value:
            value = value.replace(token, resolver())
    return value


def load_server_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """browser_mcp.json with placeholders resolved and CSKG_PYTHON applied.
    This is what both the pipeline (mcp_rdf_agent.get_mcp_client) and this
    check launch, so they cannot drift apart."""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"MCP config file not found at {path}")
    config = json.loads(path.read_text())
    servers = config.get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        raise ValueError(f"{path} has no 'mcpServers' entry")
    override = os.environ.get(PYTHON_OVERRIDE_ENV)
    for spec in servers.values():
        spec["command"] = override or _substitute(str(spec.get("command", "")))
        spec["args"] = [_substitute(str(a)) for a in spec.get("args", [])]
        # MCP stdio transports may whitelist inherited environment variables.
        # These values are stage-specific and must reach the stdio server
        # process explicitly: MCP clients may launch it with a restricted
        # environment rather than inheriting the full parent environment.
        # In particular, without CSKG_MAX_RESULT_ROWS the server silently
        # falls back to 25 rows even though the evaluator is configured for 10.
        for key in ('PROPOSAL_TELEMETRY_PATH', 'PROPOSAL_CSKG_CACHE', 'CSKG_MAX_RESULT_ROWS'):
            if os.environ.get(key):
                spec.setdefault('env', {})[key] = os.environ[key]
    return config


def server_identity(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """What a run's manifest should record about the external knowledge
    service: interpreter, script (with hash), and endpoint."""
    import hashlib
    try:
        spec = next(iter(load_server_config(config_path)["mcpServers"].values()))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    script = next((a for a in spec["args"] if a.endswith(".py")), None)
    digest = None
    if script and Path(script).exists():
        digest = hashlib.sha256(Path(script).read_bytes()).hexdigest()
    return {"command": spec["command"], "server_script": script, "server_script_sha256": digest,
            "sparql_endpoint": _arg_value(spec["args"], "--sparql-endpoint"),
            "rdf_file": _arg_value(spec["args"], "--rdf-file")}


def _arg_value(args: List[str], flag: str) -> Optional[str]:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


# --------------------------------------------------------------------------
# Offloaded-file detection
# --------------------------------------------------------------------------

def is_offloaded(path: os.PathLike) -> bool:
    """True when the file's content is not on local disk (macOS 'dataless'
    placeholder left by iCloud/File Provider sync). Reading such a file
    blocks until it downloads -- which may be never."""
    try:
        return bool(getattr(os.stat(path), "st_flags", 0) & _SF_DATALESS)
    except OSError:
        return False


def count_offloaded(root: os.PathLike, limit: int = _SCAN_LIMIT) -> Dict[str, Any]:
    """Counts offloaded files under `root` using stat only (never opens a
    file, so the scan itself cannot hang on a placeholder)."""
    seen = offloaded = 0
    example = None
    stack = [str(root)]
    while stack and seen < limit:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            seen += 1
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif getattr(entry.stat(follow_symlinks=False), "st_flags", 0) & _SF_DATALESS:
                    offloaded += 1
                    example = example or entry.path
            except OSError:
                continue
    return {"root": str(root), "files_examined": seen, "offloaded_files": offloaded,
            "example": example, "scan_complete": not stack}


def _environment_root(command: str) -> Optional[Path]:
    """The virtualenv (or prefix) an interpreter path belongs to, found
    without executing it: <env>/bin/python -> <env>."""
    p = Path(command)
    return p.parent.parent if p.parent.name in ("bin", "Scripts") else None


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def _stage(name: str, ok: bool, started: float, detail: str = "", *, error: Optional[BaseException] = None,
           remediation: Optional[str] = None, **extra) -> Dict[str, Any]:
    record = {"stage": name, "ok": ok, "seconds": round(time.monotonic() - started, 2), "detail": detail}
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error"] = str(error) or f"{type(error).__name__} (no message)"
    if remediation and not ok:
        record["remediation"] = remediation
    record.update({k: v for k, v in extra.items() if v is not None})
    return record


_OFFLOAD_FIX = (
    "Run the server with an interpreter whose environment is fully on local disk and outside any "
    "cloud-synced folder: set \"command\": \"${CURRENT_PYTHON}\" in browser_mcp.json (uses the "
    "interpreter running the pipeline), or export CSKG_PYTHON=/path/to/python."
)


def _explain_hang(command: str) -> Dict[str, Any]:
    """After a timeout: is the interpreter's environment offloaded?"""
    root = _environment_root(command)
    if not root or not root.exists():
        return {}
    scan = count_offloaded(root)
    return {"offloaded_scan": scan} if scan["offloaded_files"] else {}


def check_interpreter(command: str, timeout: float = 20.0) -> Dict[str, Any]:
    started = time.monotonic()
    path = Path(command)
    if not path.exists():
        return _stage("interpreter", False, started, f"{command} does not exist",
                      remediation="Fix \"command\" in browser_mcp.json (or set CSKG_PYTHON).")
    if is_offloaded(path):
        return _stage("interpreter", False, started, f"{command} is offloaded (not downloaded to this disk)",
                      remediation=_OFFLOAD_FIX)
    try:
        out = subprocess.run([command, "-c", "import sys; print(sys.version.split()[0]); print(sys.prefix)"],
                             capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        return _stage("interpreter", False, started, f"{command} did not start within {timeout:.0f}s",
                      error=exc, remediation=_OFFLOAD_FIX, **_explain_hang(command))
    except OSError as exc:
        return _stage("interpreter", False, started, f"{command} could not be executed", error=exc,
                      remediation="Fix \"command\" in browser_mcp.json (or set CSKG_PYTHON).")
    if out.returncode != 0:
        return _stage("interpreter", False, started, f"exit code {out.returncode}: {out.stderr.strip()[-400:]}")
    version, prefix = (out.stdout.strip().splitlines() + ["", ""])[:2]
    return _stage("interpreter", True, started, f"Python {version} at {prefix}", prefix=prefix)


def check_imports(command: str, timeout: float = 90.0) -> Dict[str, Any]:
    started = time.monotonic()
    code = ("import importlib, sys\n"
            f"for name in {SERVER_IMPORTS!r}:\n"
            "    try:\n        importlib.import_module(name)\n"
            "    except Exception as exc:\n"
            "        print(f'IMPORT_FAILED {name}: {type(exc).__name__}: {exc}'); sys.exit(3)\n"
            f"for name in {OPTIONAL_SERVER_IMPORTS!r}:\n"
            "    try:\n        importlib.import_module(name)\n"
            "    except Exception:\n        print(f'OPTIONAL_MISSING {name}')\n"
            "print('IMPORTS_OK')\n")
    try:
        out = subprocess.run([command, "-c", code], capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        extra = _explain_hang(command)
        scan = extra.get("offloaded_scan")
        detail = f"importing the server's dependencies did not finish within {timeout:.0f}s"
        if scan:
            detail += f"; {scan['offloaded_files']} file(s) under {scan['root']} are offloaded (not on local disk)"
        return _stage("imports", False, started, detail, error=exc, remediation=_OFFLOAD_FIX, **extra)
    if "IMPORTS_OK" in out.stdout:
        missing = [l.split(" ", 1)[1] for l in out.stdout.splitlines() if l.startswith("OPTIONAL_MISSING ")]
        detail = ", ".join(SERVER_IMPORTS)
        if missing:
            detail += f"; NOT installed (optional, changes behaviour): {', '.join(missing)}"
        return _stage("imports", True, started, detail, missing_optional=missing or None)
    failed = next((l for l in out.stdout.splitlines() if l.startswith("IMPORT_FAILED")), out.stderr.strip()[-400:])
    return _stage("imports", False, started, failed,
                  remediation="Install the server's dependencies into that interpreter "
                              "(mcp, rdflib[sparql], tiktoken, openai, python-dotenv).")


def check_server_script(args: List[str]) -> Dict[str, Any]:
    started = time.monotonic()
    script = next((a for a in args if a.endswith(".py")), None)
    if not script:
        return _stage("server_script", False, started, "no .py script among the server args")
    if not Path(script).exists():
        return _stage("server_script", False, started, f"{script} does not exist",
                      remediation="Fix the script path in browser_mcp.json (\"${PROJECT_ROOT}\" is accepted).")
    if is_offloaded(script):
        return _stage("server_script", False, started, f"{script} is offloaded (not downloaded to this disk)",
                      remediation="Download the file locally or move the project out of the synced folder.")
    return _stage("server_script", True, started, script)


def check_sparql_endpoint(args: List[str], timeout: float = 30.0) -> Dict[str, Any]:
    started = time.monotonic()
    endpoint = _arg_value(args, "--sparql-endpoint")
    if not endpoint:
        rdf = _arg_value(args, "--rdf-file")
        if rdf:
            return _stage("sparql_endpoint", True, started, f"local file mode ({rdf}); no endpoint to reach")
        return _stage("sparql_endpoint", False, started, "neither --sparql-endpoint nor --rdf-file is configured")
    # The server verifies TLS against the OS trust store (truststore); do
    # the same here, but with a context local to this one request -- this
    # function also runs inside the evaluation process, whose Neo4j and
    # OpenAI connections must not have their TLS handling swapped globally.
    context = None
    try:
        import ssl
        import truststore
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        pass
    url = endpoint + ("&" if "?" in endpoint else "?") + urllib.parse.urlencode(
        {"query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"})
    request = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            body = response.read(4096)
            status, final = response.status, response.geturl()
    except Exception as exc:
        if _arg_value(args, "--rdf-file"):
            return _stage("sparql_endpoint", False, started,
                          f"{endpoint} did not answer a trivial query (hybrid mode: ATT&CK is served from the "
                          f"local file; only the CVE/CVSS tools are affected)", error=exc,
                          remediation="External service. Check the network path and retry later.")
        return _stage("sparql_endpoint", False, started, f"{endpoint} did not answer a trivial query", error=exc,
                      remediation="External service. Check the network path and retry later; the server also "
                                  "supports a local copy via --rdf-file.")
    ok = status == 200 and b"bindings" in body
    return _stage("sparql_endpoint", ok, started,
                  f"HTTP {status} from {final}" + ("" if ok else "; response was not a SPARQL JSON result"))


async def _session_stages(spec: Dict[str, Any], *, handshake_timeout: float, call_timeout: float,
                          lookup: bool) -> List[Dict[str, Any]]:
    """handshake -> tools -> lookup against a freshly started server."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    stages: List[Dict[str, Any]] = []
    params = StdioServerParameters(command=spec["command"], args=spec["args"], env=spec.get("env"))

    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as errlog:
        def stderr_tail(lines: int = 12) -> str:
            errlog.flush()
            errlog.seek(0)
            return "\n".join(errlog.read().strip().splitlines()[-lines:])

        started = time.monotonic()
        current = "handshake"
        try:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), handshake_timeout)
                    stages.append(_stage("handshake", True, started, "MCP initialize completed"))

                    started, current = time.monotonic(), "tools"
                    listing = await asyncio.wait_for(session.list_tools(), call_timeout)
                    names = {t.name for t in listing.tools}
                    missing = [t for t in REQUIRED_TOOLS if t not in names]
                    stages.append(_stage("tools", not missing, started,
                                         f"{len(names)} tools registered" + (f"; missing {missing}" if missing else "")))
                    if missing or not lookup:
                        return stages

                    started, current = time.monotonic(), "lookup"
                    technique = _tool_text(await asyncio.wait_for(session.call_tool(
                        "get_technique_by_id", {"technique_id": PROBE_TECHNIQUE_ID, "include_description": False}),
                        call_timeout))
                    mapping = _tool_text(await asyncio.wait_for(session.call_tool(
                        "get_technique_mitigations_by_keyword", {"keywords": [
                            PROBE_TECHNIQUE_NAME, "Network Service Scanning", "Active Scanning"]}), call_timeout))
                    problems = []
                    if PROBE_TECHNIQUE_NAME.lower() not in technique.lower():
                        problems.append(f"get_technique_by_id({PROBE_TECHNIQUE_ID}) returned {technique[:160]!r}")
                    if (PROBE_TECHNIQUE_NAME.lower() not in mapping.lower() or
                            not any(word in mapping.lower() for word in ("mitigation", "filter network traffic"))):
                        problems.append(f"get_technique_mitigations_by_keyword returned {mapping[:160]!r}")
                    stages.append(_stage(
                        "lookup", not problems, started,
                        "; ".join(problems) or f"{PROBE_TECHNIQUE_ID} resolved to '{PROBE_TECHNIQUE_NAME}' with "
                                               f"{len(mapping.splitlines())} technique/mitigation row(s)",
                        remediation="The server runs but the knowledge graph returned no data; check the SPARQL "
                                    "endpoint's contents."))
        except BaseException as exc:  # includes TimeoutError and anyio's exception groups
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            done = {s["stage"] for s in stages}
            if current not in done:
                root = _root_cause(exc)
                timed_out = isinstance(root, (asyncio.TimeoutError, TimeoutError))
                tail = stderr_tail()
                last_line = _key_stderr_line(tail)
                if timed_out:
                    detail = (f"the server process started but did not answer within "
                              f"{handshake_timeout if current == 'handshake' else call_timeout:.0f}s")
                else:
                    detail = "the server process exited" + (f": {last_line[:300]}" if last_line else "")
                extra = _explain_hang(spec["command"]) if timed_out else {}
                scan = extra.get("offloaded_scan")
                if scan:
                    detail += f"; {scan['offloaded_files']} file(s) under {scan['root']} are offloaded (not on local disk)"
                stages.append(_stage(
                    current, False, started, detail, error=root, server_stderr=tail or None,
                    remediation=_OFFLOAD_FIX if scan else (
                        "Run the configured command by hand to see where it stops; a server that starts but "
                        "never answers is usually blocked importing its dependencies or reaching its SPARQL endpoint."
                        if timed_out else
                        "See server_stderr. The server connects to its SPARQL endpoint during startup, so an "
                        "unreachable endpoint or a missing dependency ends the process here."),
                    **extra))
    return stages


def _key_stderr_line(tail: str) -> str:
    """The most telling line of the server's stderr: the first one that
    reports an error (later ones are usually generic teardown noise),
    without its logging prefix."""
    import re
    lines = [l.strip() for l in (tail or "").splitlines() if l.strip()]
    if not lines:
        return ""
    chosen = next((l for l in lines if re.search(r"\bERROR\b|Error\b|Exception\b", l)), lines[-1])
    return re.sub(r"^\d{4}-\d{2}-\d{2} [\d:,.]+ - \S+ - \w+ - ", "", chosen)


def _diagnose(stages: List[Dict[str, Any]], failed: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Connects stages that each look harmless alone. rdflib reports ANY
    failure to reach a SPARQL endpoint as "You did something wrong
    formulating either the URI or your SPARQL query", so the server's own
    message cannot tell a certificate problem from a bad query."""
    by_name = {s["stage"]: s for s in stages}
    no_truststore = "truststore" in (by_name.get("imports", {}).get("missing_optional") or [])
    endpoint = by_name.get("sparql_endpoint", {})
    stderr = str((failed or {}).get("server_stderr") or "") + str((failed or {}).get("detail") or "")
    # Only a certificate error is attributed to truststore; any other
    # endpoint failure (outage, proxy, DNS) is left as reported.
    cert_error = any("certificate verify failed" in text.lower() for text in (str(endpoint.get("error", "")), stderr))
    endpoint_stage = failed and failed["stage"] in ("handshake", "lookup") and (
        "SPARQL endpoint" in stderr or cert_error)
    if endpoint_stage and cert_error:
        cause = ("the SPARQL endpoint's TLS certificate could not be verified with this Python's default CA store"
                 + (", and `truststore` (which makes the server use the operating system's trust store instead) "
                    "is not installed in the server's interpreter." if no_truststore else "."))
        fix = ("Add truststore to the project so every environment built from uv.lock has it: "
               "`uv add truststore` (with UV_PROJECT_ENVIRONMENT pointing at the environment in use), "
               "then rerun this check." if no_truststore else
               "Check the system clock and the machine's trust store; the endpoint's certificate chain is not accepted.")
        return {"cause": cause, "fix": fix}
    return None


def _root_cause(exc: BaseException) -> BaseException:
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return exc


def _tool_text(result: Any) -> str:
    return "\n".join(getattr(c, "text", "") or "" for c in getattr(result, "content", []) or [])


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

async def run_checks_async(config_path: Optional[Path] = None, *, deep: bool = True,
                           handshake_timeout: Optional[float] = None,
                           call_timeout: Optional[float] = None) -> Dict[str, Any]:
    """`deep=True` (CLI, evaluation preflight): every stage, generous
    timeouts. `deep=False` (per-process readiness probe inside the
    pipeline): config, interpreter, script and handshake/tools only, with
    short timeouts so the result lands inside one retrieval deadline."""
    stages: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "mode": "full" if deep else "readiness", "stages": stages}

    def finish() -> Dict[str, Any]:
        # `sparql_endpoint` is advisory: it probes the endpoint from this
        # process, which can differ from the server's own network/TLS path.
        # The server's handshake + lookup are what decide the outcome.
        failed = next((s for s in stages if not s["ok"] and not s.get("advisory")), None)
        warnings = [f"{s['stage']}: {s['detail']}" + (f" [{s['error_type']}: {s['error']}]" if s.get("error_type") else "")
                    for s in stages if not s["ok"] and s.get("advisory")]
        report["status"] = "failed" if failed else "ok"
        report["failed_stage"] = failed["stage"] if failed else None
        report["warnings"] = warnings
        diagnosis = _diagnose(stages, failed)
        if diagnosis:
            report["diagnosis"] = diagnosis
            if failed:
                failed["remediation"] = diagnosis["fix"]
        if failed:
            report["summary"] = (f"CSKG dependency failure at stage '{failed['stage']}': {failed['detail']}"
                                 + (f" [{failed['error_type']}: {failed['error']}]" if failed.get("error_type") else "")
                                 + (f" (also: {'; '.join(warnings)})" if warnings else "")
                                 + (f" Likely cause: {diagnosis['cause']}" if diagnosis else ""))
        else:
            report["summary"] = "CSKG server healthy" + (f" (warning -- {'; '.join(warnings)})" if warnings else "")
        return report

    started = time.monotonic()
    try:
        config = load_server_config(config_path)
        name, spec = next(iter(config["mcpServers"].items()))
    except Exception as exc:
        stages.append(_stage("config", False, started, "browser_mcp.json could not be used", error=exc,
                             remediation="Create browser_mcp.json from browser_mcp.json.example."))
        return finish()
    report["server"] = {"name": name, "command": spec["command"], "args": spec["args"]}
    stages.append(_stage("config", True, started, f"server '{name}'"))

    stages.append(await asyncio.to_thread(check_interpreter, spec["command"], 20.0 if deep else 10.0))
    if not stages[-1]["ok"]:
        return finish()
    stages.append(check_server_script(spec["args"]))
    if not stages[-1]["ok"]:
        return finish()
    if deep:
        stages.append(await asyncio.to_thread(check_imports, spec["command"]))
        if not stages[-1]["ok"]:
            return finish()
        # Reported, but not a reason to stop: the handshake below tells
        # whether the server itself can reach the endpoint.
        stages.append({**await asyncio.to_thread(check_sparql_endpoint, spec["args"]), "advisory": True})
    stages.extend(await _session_stages(
        spec, handshake_timeout=handshake_timeout or (90.0 if deep else 35.0),
        call_timeout=call_timeout or (60.0 if deep else 10.0), lookup=deep))
    return finish()


def run_checks(config_path: Optional[Path] = None, **kwargs) -> Dict[str, Any]:
    """Synchronous wrapper; safe to call whether or not an event loop is running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_checks_async(config_path, **kwargs))
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(run_checks_async(config_path, **kwargs))).result()


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"CSKG health check ({report.get('mode')}) at {report.get('checked_at')}: {report['status'].upper()}"]
    server = report.get("server") or {}
    if server:
        lines.append(f"  command: {server.get('command')}")
        lines.append(f"  args:    {' '.join(server.get('args', []))}")
    for s in report["stages"]:
        mark = "ok" if s["ok"] else ("warn" if s.get("advisory") else "FAILED")
        lines.append(f"  [{mark:>6}] {s['stage']:<16} {s['seconds']:>6.2f}s  {s['detail']}")
        if not s["ok"]:
            if s.get("error_type"):
                lines.append(f"           error: {s['error_type']}: {s['error']}")
            scan = s.get("offloaded_scan")
            if scan:
                lines.append(f"           offloaded: {scan['offloaded_files']} of {scan['files_examined']} entries under "
                             f"{scan['root']} (e.g. {scan['example']})")
            if s.get("server_stderr"):
                lines.append("           server stderr (last lines):")
                lines += [f"             {l}" for l in s["server_stderr"].splitlines()]
            if s.get("remediation"):
                lines.append(f"           fix: {s['remediation']}")
    lines.append(report["summary"])
    if report.get("diagnosis"):
        lines.append(f"Fix: {report['diagnosis']['fix']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Independent CSKG MCP server health check")
    parser.add_argument("--config", type=Path, default=None, help="Defaults to browser_mcp.json in the project root")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    parser.add_argument("--readiness", action="store_true", help="Quick probe only (what the pipeline runs)")
    args = parser.parse_args(argv)
    report = run_checks(args.config, deep=not args.readiness)
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
