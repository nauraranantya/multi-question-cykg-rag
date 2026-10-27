#!/usr/bin/env python3
"""
MITRE ATT&CK SPARQL MCP Server with Local/Remote Support

This server provides tools to query MITRE ATT&CK data using SPARQL queries
against either local RDF files or remote SPARQL endpoints based on the MITRE ATT&CK ontology.
"""

from typing import Any, Dict, List, Optional
import os
import argparse
import json
import re
import socket
import sys
import time
import tiktoken
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

# This subprocess is launched (via browser_mcp.json) using the main
# project's own .venv, so it normally inherits OPENAI_API_KEY from the
# parent process's already-loaded environment (src/config/settings.py's
# load_dotenv() runs before this subprocess is ever spawned). Loaded here
# too, defensively, so this script also works standalone (e.g. run
# directly for testing) without depending on that inheritance.
from dotenv import load_dotenv
load_dotenv()

# rdflib's SPARQLConnector.query() calls urllib's urlopen() with no timeout
# on any code path (verified directly in its source) -- a slow/stuck
# endpoint (e.g. the w3id.org -> http://sepses.ifs.tuwien.ac.at redirect
# chain measured taking 30s+ on some queries) hangs the request forever,
# freezing the whole agent loop with no recovery (observed directly: an
# 11+ minute hang on a single CallToolRequest). Passing timeout= to
# SPARQLStore's constructor doesn't help -- it's never extracted from
# self.kwargs and threaded through to urlopen(). Setting the process-wide
# default socket timeout is the only lever that actually reaches these
# calls; every socket operation in this process that doesn't set its own
# timeout will raise socket.timeout after this many seconds instead of
# blocking indefinitely.
socket.setdefaulttimeout(30)

# Some SPARQL endpoints (e.g. the public SEPSES endpoint at w3id.org) present
# a certificate chain that isn't in every installed certifi snapshot, even
# though it validates fine against the OS's own trust store (verified: curl
# and macOS both accept it). truststore patches ssl to verify against the OS
# trust store instead of certifi's bundled one -- must run before rdflib's
# SPARQLStore (via urllib) opens any HTTPS connection.
try:
    import truststore
    truststore.inject_into_ssl()
    HAS_TRUSTSTORE = True
except ImportError:
    HAS_TRUSTSTORE = False  # reported once logging is configured, below

import rdflib
from openai import OpenAI
# The module is next to this script for both direct and test-loader execution.
import importlib.util as _importlib_util
_eval_spec = _importlib_util.spec_from_file_location("cskg_evaluation_support", os.path.join(os.path.dirname(__file__), "evaluation_support.py"))
_eval_support = _importlib_util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_support)
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base

# Configure logging
logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]
    )

# Check for SPARQLStore availability
try:
    from rdflib.plugins.stores.sparqlstore import SPARQLStore
    HAS_SPARQLSTORE = True
except ImportError:
    HAS_SPARQLSTORE = False
    logger.warning("SPARQLStore not available. SPARQL Endpoint Mode will be disabled.")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="MITRE ATT&CK SPARQL MCP Server v1.0.0")
parser.add_argument("--rdf-file", default="", help="Path to the local RDF file containing MITRE ATT&CK data")
parser.add_argument("--sparql-endpoint", default="", help="SPARQL endpoint URL (empty for Local File Mode)")
# Hybrid mode: pass BOTH. ATT&CK queries then run on the local file and
# CVE/CVSS/CWE/CAPEC queries (anything using those vocabularies) on the
# endpoint; an unreachable endpoint only disables the CVE tools instead of
# stopping the server. See mcp-cskg-rdf/build_attack_rdf.py for why.
args = parser.parse_args()

logger.info("Starting MITRE ATT&CK SPARQL MCP Server v1.0.0")
if not HAS_TRUSTSTORE:
    # Previously skipped in silence. Without it this process verifies HTTPS
    # against Python's default CA store, which the SEPSES endpoint is known
    # to fail on some builds -- and rdflib then reports that as a malformed
    # query (see _describe_error below).
    logger.warning("truststore is not installed: HTTPS certificates are verified against Python's default CA "
                   "store instead of the operating system's trust store. Install it (it is listed in "
                   "mcp-cskg-rdf/requirements.txt) if the SPARQL endpoint fails certificate verification.")


def _describe_error(exc: BaseException) -> str:
    """rdflib's SPARQLConnector turns ANY failure to reach the endpoint --
    DNS, timeout, TLS verification -- into ValueError("You did something
    wrong formulating either the URI or your SPARQL query"). The real
    error survives only as the exception's context; include it."""
    cause = exc.__cause__ or exc.__context__
    return f"{exc}" + (f" (underlying error: {type(cause).__name__}: {cause})" if cause else "")

# Define namespaces
ATTACK = rdflib.Namespace("http://w3id.org/sepses/vocab/ref/attack#")
CAPEC = rdflib.Namespace("http://w3id.org/sepses/vocab/ref/capec#")
RDF = rdflib.Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")

# Initialize FastMCP server
mcp = FastMCP(
    "MITRE ATT&CK SPARQL",
    dependencies=["rdflib[sparql]"],
    lifespan=lambda mcp: attack_triplestore_lifespan(mcp, args.rdf_file, args.sparql_endpoint)
)

@asynccontextmanager
async def attack_triplestore_lifespan(server: FastMCP, rdf_file: str, sparql_endpoint: str) -> AsyncIterator[Dict[str, Any]]:
    """Manage the lifespan of the MITRE ATT&CK triplestore.

    Args:
        server (FastMCP): The FastMCP server instance.
        rdf_file (str): Path to the local RDF file.
        sparql_endpoint (str): URL of the SPARQL endpoint, if any.

    Yields:
        Dict[str, Any]: Context dictionary containing the graph and configuration.
    """
    logger.info(f"Initializing MITRE ATT&CK triplestore with rdf_file={rdf_file}, sparql_endpoint={sparql_endpoint}")

    metrics = {"queries": 0, "total_time": 0.0}
    max_tokens = 10000
    local_graph = remote_graph = None
    remote_error = None
    file_path = ""

    if rdf_file:
        local_graph = rdflib.Graph()
        file_path = os.path.join(os.path.dirname(__file__), rdf_file)
        logger.info(f"Loading local RDF file: {file_path}")
        try:
            local_graph.parse(file_path, format="turtle")
            logger.info(f"Loaded {len(local_graph)} triples from local file")
        except FileNotFoundError:
            logger.error(f"RDF file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load RDF file: {str(e)}")
            raise

    if sparql_endpoint and HAS_SPARQLSTORE:
        logger.info(f"Connecting to SPARQL endpoint: {sparql_endpoint}")
        try:
            remote_graph = SPARQLStore(query_endpoint=sparql_endpoint)
            # Test connection
            remote_graph.query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
            logger.info(f"Successfully connected to {sparql_endpoint}")
        except Exception as e:
            remote_error = _describe_error(e)
            if local_graph is None:
                logger.error(f"Failed to connect to SPARQL endpoint: {remote_error}")
                raise
            # Hybrid mode: ATT&CK still works from the local file.
            logger.warning(f"SPARQL endpoint unavailable, CVE tools disabled until restart: {remote_error}")
            remote_graph = None

    if local_graph is None and remote_graph is None:
        raise ValueError("No knowledge graph configured: pass --rdf-file, --sparql-endpoint, or both.")

    try:
        local = _eval_support.CachedGraph(local_graph, str(file_path)) if local_graph is not None else None
        remote = _eval_support.CachedGraph(remote_graph, sparql_endpoint) if remote_graph is not None else None
        # "graph" answers ATT&CK queries; "cve_graph" answers CVE-family
        # queries (None = configured but unreachable, see remote_error).
        graph = local or remote
        if local is not None and sparql_endpoint:
            cve_graph = remote
        else:
            cve_graph = graph
        logger.info("MITRE ATT&CK triplestore initialized successfully")
        yield {
            "graph": graph,
            "cve_graph": cve_graph,
            "cve_error": remote_error,
            "metrics": metrics,
            "max_tokens": max_tokens,
            "rdf_file": rdf_file,
            "sparql_endpoint": sparql_endpoint,
            # True when ATT&CK queries go to the remote endpoint (picks the
            # cheaper statistics query); False when a local file is loaded.
            "is_sparql_endpoint": local is None,
        }
    finally:
        logger.info("Shutting down MITRE ATT&CK triplestore connection")
        if remote_graph is not None:
            try:
                remote_graph.close()
            except Exception:
                pass

MAX_DESCRIPTION_CHARS = 400
MAX_RESULT_CHARS = 6000  # ~1500 tokens -- keeps a single tool call's context contribution bounded
MAX_RESULT_ROWS = max(1, int(os.environ.get("CSKG_MAX_RESULT_ROWS", "25")))


def _top_level_tokens(query: str):
    """Yield SPARQL keyword spans outside strings, comments, IRIs and groups."""
    depth = 0
    i = 0
    while i < len(query):
        char = query[i]
        if char == "#":
            newline = query.find("\n", i)
            i = len(query) if newline < 0 else newline + 1
            continue
        if char in "'\"":
            marker = char * (3 if query.startswith(char * 3, i) else 1)
            i += len(marker)
            while i < len(query):
                if query[i] == "\\":
                    i += 2
                elif query.startswith(marker, i):
                    i += len(marker)
                    break
                else:
                    i += 1
            continue
        if char == "<":
            close = query.find(">", i + 1)
            whitespace = [p for p in (query.find(" ", i + 1), query.find("\n", i + 1), query.find("\t", i + 1)) if p >= 0]
            if close >= 0 and close < min(whitespace, default=len(query)):
                i = close + 1
                continue
        if char == "{":
            depth += 1
            i += 1
            continue
        if char == "}":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and (char.isalpha() or char == "_"):
            start = i
            i += 1
            while i < len(query) and (query[i].isalnum() or query[i] in "_-"):
                i += 1
            yield query[start:i].upper(), start, i
            continue
        i += 1


def cap_select_query(query: str, max_rows: int = MAX_RESULT_ROWS) -> str:
    """Apply a hard outer-row cap before a SELECT reaches the SPARQL endpoint.

    Formatting alone is too late for SPARQLStore: it reads the full HTTP
    response before returning a Result. Existing smaller LIMITs are kept;
    larger ones are reduced. Nested-subquery LIMITs do not count as the
    outer limit.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    form_match = re.match(
        r"(?is)^\s*(?:(?:\#[^\r\n]*(?:\r?\n|$))|(?:PREFIX\s+[\w-]*:\s*<[^>]*>|BASE\s*<[^>]*>)\s*)*(SELECT|ASK|CONSTRUCT|DESCRIBE)\b",
        query,
    )
    if not form_match or form_match.group(1).upper() != "SELECT":
        return query
    tokens = list(_top_level_tokens(query))

    limit = next(((start, end) for token, start, end in tokens if token == "LIMIT"), None)
    if limit:
        _, end = limit
        match = re.match(r"\s*(\d+)", query[end:])
        if match:
            current = int(match.group(1))
            if current > max_rows:
                start = end + match.start(1)
                stop = end + match.end(1)
                return query[:start] + str(max_rows) + query[stop:]
            return query

    offset = next((start for token, start, _ in tokens if token == "OFFSET"), None)
    if offset is not None:
        insertion = offset
    else:
        # Keep an optional final semicolon after the new solution modifier.
        end = len(query.rstrip())
        insertion = end - 1 if end and query.rstrip().endswith(";") else end
    return query[:insertion].rstrip() + f"\nLIMIT {max_rows}\n" + query[insertion:]


def format_sparql_results(results, include_description: bool = False) -> str:
    """Format SPARQL query results into a readable string.

    Bounded on two axes, independent of whether the underlying SPARQL query
    itself has a LIMIT (many of the ~40 tools don't): each `description`
    field is truncated to MAX_DESCRIPTION_CHARS, and the total formatted
    output is truncated to MAX_RESULT_CHARS (by whole result rows, never
    mid-row) with a note about how many rows were omitted. Without this, a
    broad keyword/tactic match can return dozens of full-paragraph
    descriptions in one tool call, ballooning the agent's context and
    latency for no benefit -- observed directly, not just a theoretical risk.

    Args:
        results: SPARQL query results
        include_description: Whether to include descriptions (if available)

    Returns:
        Formatted string representation of the results
    """
    if not results:
        return "No results found."

    formatted_results = []
    rows_truncated = False

    for row_index, row in enumerate(results):
        # Do not materialize or serialize a broad result set just to throw
        # most of it away at the character limit. Consume one extra row only
        # to distinguish an exact-size result from a truncated one.
        if row_index >= MAX_RESULT_ROWS:
            rows_truncated = True
            break
        result_parts = []

        # Handle different variable bindings
        for var_name, value in row.asdict().items():
            if value:
                # Clean up the value representation
                if isinstance(value, rdflib.URIRef):
                    # Extract the local name for cleaner display
                    local_name = str(value).split('#')[-1] if '#' in str(value) else str(value).split('/')[-1]
                    result_parts.append(f"{var_name}: {local_name}")
                else:
                    text = str(value)
                    if var_name == "description" and len(text) > MAX_DESCRIPTION_CHARS:
                        text = text[:MAX_DESCRIPTION_CHARS].rstrip() + "... [truncated]"
                    result_parts.append(f"{var_name}: {text}")

        formatted_results.append(" | ".join(result_parts))

    output = "\n".join(formatted_results)
    if len(output) > MAX_RESULT_CHARS:
        kept, running = [], 0
        for line in formatted_results:
            if running + len(line) + 1 > MAX_RESULT_CHARS:
                break
            kept.append(line)
            running += len(line) + 1
        omitted = len(formatted_results) - len(kept)
        truncation_note = (
            f"\n... [{omitted} more result(s) truncated -- narrow your query "
            f"(e.g. a more specific keyword/tactic, or include_description=False) to see more"
        )
        if rows_truncated:
            truncation_note += f"; hard row cap is {MAX_RESULT_ROWS}"
        output = "\n".join(kept) + truncation_note + "]"
    elif rows_truncated:
        output += f"\n... [results capped at {MAX_RESULT_ROWS} rows; narrow the query to see a more specific subset]"
    return output

#################################################################
# Core Infrastructure Tools
#################################################################
# Tools
@mcp.tool()
def set_max_tokens(tokens: int, ctx: Context) -> str:
    """Set the maximum token limit for prompts.

    Args:
        tokens (int): The new maximum token limit (must be positive).
        ctx (Context): The FastMCP context object.

    Returns:
        str: Confirmation message or error if the value is invalid.
    """
    if tokens <= 0:
        return "Error: MAX_TOKENS must be positive."
    ctx.request_context.lifespan_context["max_tokens"] = tokens
    logger.info(f"Set MAX_TOKENS to {tokens}")
    return f"MAX_TOKENS set to {tokens}"

_CVE_FAMILY = re.compile(r"w3id\.org/sepses/vocab/ref/(?:cve|cvss|cwe|capec|cpe|icsa)#")
_ATTACK_VOCAB = re.compile(r"w3id\.org/sepses/vocab/ref/attack#")


def _graph_for(query: str, lifespan: Dict[str, Any]):
    """In hybrid mode, CVE-family queries go to the endpoint and everything
    else (ATT&CK) to the local file. A query mixing both vocabularies stays
    local: the two sources cannot be joined in one query."""
    if lifespan.get("cve_graph") is lifespan["graph"]:
        return lifespan["graph"]
    if _CVE_FAMILY.search(query) and not _ATTACK_VOCAB.search(query):
        return lifespan.get("cve_graph")
    return lifespan["graph"]


@mcp.tool()
def execute_sparql_query(query: str, ctx: Context, include_description: bool = False) -> str:
    """Execute a custom SPARQL query against the MITRE ATT&CK knowledge graph.
    
    Args:
        query: SPARQL query string to execute
        ctx: FastMCP context object
        include_description: Whether to include descriptions in results (default: False)
        
    Returns:
        Formatted query results
    """
    lifespan = ctx.request_context.lifespan_context
    graph = _graph_for(query, lifespan)
    if graph is None:
        return ("Error executing SPARQL query: the CVE/CVSS knowledge base "
                f"({lifespan['sparql_endpoint']}) is unreachable: {lifespan.get('cve_error')}")
    start_time = time.time()
    bounded_query = cap_select_query(query)

    try:
        results = graph.query(bounded_query)
        ctx.request_context.lifespan_context["metrics"]["queries"] += 1
        ctx.request_context.lifespan_context["metrics"]["total_time"] += time.time() - start_time
        logger.info(bounded_query)
        return format_sparql_results(results, include_description)
    except Exception as e:
        logger.error(f"SPARQL query error: {_describe_error(e)}")
        return f"Error executing SPARQL query: {_describe_error(e)}"

@mcp.tool()
def get_server_mode(ctx: Context) -> str:
    """Get the current mode of the MITRE ATT&CK server.
    
    Args:
        ctx: FastMCP context object
        
    Returns:
        A message indicating the mode and data source
    """
    rdf_file = ctx.request_context.lifespan_context["rdf_file"]
    sparql_endpoint = ctx.request_context.lifespan_context["sparql_endpoint"]
    is_sparql_endpoint = ctx.request_context.lifespan_context["is_sparql_endpoint"]
    
    if rdf_file and sparql_endpoint:
        cve_state = "reachable" if ctx.request_context.lifespan_context.get("cve_graph") is not None else \
            f"UNREACHABLE ({ctx.request_context.lifespan_context.get('cve_error')})"
        return (f"Hybrid Mode: ATT&CK from local file '{rdf_file}'; CVE/CVSS from endpoint "
                f"'{sparql_endpoint}' ({cve_state})")
    if is_sparql_endpoint:
        return f"SPARQL Endpoint Mode with Endpoint: '{sparql_endpoint}'"
    else:
        return f"Local File Mode with Dataset: '{rdf_file or 'empty graph'}'"

@mcp.tool()
def get_attack_statistics(ctx: Context) -> str:
    """Get statistical summary of the MITRE ATT&CK knowledge base.
    
    Args:
        ctx: FastMCP context object
        
    Returns:
        JSON string containing statistics about the knowledge base
    """
    graph = ctx.request_context.lifespan_context["graph"]
    # One grouped count. (The previous query combined one OPTIONAL per class,
    # which multiplies the class sizes together -- it never finished on a
    # real local graph.)
    classes = {
        "Technique": "techniqueCount", "SubTechnique": "subtechniqueCount",
        "AdversaryGroup": "groupCount", "Software": "softwareCount", "Malware": "malwareCount",
        "Mitigation": "mitigationCount", "Tactic": "tacticCount", "Asset": "assetCount",
        "DataSource": "dataSourceCount", "DataComponent": "dataComponentCount",
    }
    values = " ".join(f"attack:{name}" for name in classes)
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    SELECT ?type (COUNT(DISTINCT ?s) AS ?n) WHERE {{
        VALUES ?type {{ {values} }}
        ?s a ?type .
    }}
    GROUP BY ?type
    """
    try:
        stats = {key: 0 for key in classes.values()}
        for row in graph.query(query):
            stats[classes[str(row.type).rsplit("#", 1)[-1]]] = int(row.n)
        return json.dumps(stats, indent=2)
    except Exception as e:
        logger.error(f"Statistics query error: {str(e)}")
        return f"Error retrieving statistics: {str(e)}"

@mcp.tool()
def health_check(ctx: Context) -> str:
    """Check the health of the MITRE ATT&CK triplestore connection.
    
    Args:
        ctx: FastMCP context object
        
    Returns:
        Health status message
    """
    lifespan = ctx.request_context.lifespan_context
    graph = lifespan["graph"]
    try:
        # Simple test query
        results = list(graph.query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"))
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return f"Unhealthy: {str(e)}"
    if lifespan.get("cve_graph") is graph:
        return "Healthy - MITRE ATT&CK triplestore is responsive"
    if lifespan.get("cve_graph") is None:
        return ("Degraded - MITRE ATT&CK (local file) is responsive; the CVE/CVSS endpoint is unreachable: "
                f"{lifespan.get('cve_error')}")
    try:
        list(lifespan["cve_graph"].query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"))
        return "Healthy - MITRE ATT&CK (local file) and the CVE/CVSS endpoint are responsive"
    except Exception as e:
        return f"Degraded - MITRE ATT&CK (local file) is responsive; the CVE/CVSS endpoint failed: {_describe_error(e)}"

#################################################################
# Technique Query Tools
#################################################################

@mcp.tool()
def get_all_techniques(ctx: Context,  include_description: bool = False) -> str:
    """Get all techniques in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?label ?description WHERE {
        ?technique a attack:Technique .
        ?technique dcterm:title ?label .
        OPTIONAL { ?technique dcterm:description ?description }  
    }
    ORDER BY ?label
    """
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_techniques_by_keyword(ctx: Context,  keyword: str, include_description: bool = False) -> str:
    """Get all techniques in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?label ?description WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?label .
        OPTIONAL {{ ?technique dcterm:description ?description }}  
        
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}")) ||
            CONTAINS(LCASE(?description), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    LIMIT 50
    """
    return execute_sparql_query(query, ctx, include_description)


@mcp.tool()
def get_technique_mitigations_by_keyword(keywords: List[str], ctx: Context) -> str:
    """Find ATT&CK techniques matching a behavior keyword and return their mitigations.

    This is a single bounded graph lookup for technique-mapping questions. It
    avoids an extra model-powered text-to-SPARQL call and keeps both the
    technique IDs/titles and mitigation titles grounded in the same result.

    Args:
        keywords: Up to three concise behavior terms or technique phrases to
            match against ATT&CK technique titles and descriptions.
        ctx: FastMCP context object.
    """
    terms = [term.strip() for term in keywords if term and term.strip()][:3]
    if not terms:
        return "No results found."
    title_filters = [f"CONTAINS(LCASE(STR(?techniqueLabel)), LCASE({json.dumps(term)}))" for term in terms]
    description_filters = [f"CONTAINS(LCASE(STR(?description)), LCASE({json.dumps(term)}))" for term in terms]
    matches = " ||\n            ".join(title_filters + description_filters)
    title_matches = " || ".join(title_filters)
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>

    SELECT DISTINCT ?techniqueId ?techniqueLabel ?mitigationLabel WHERE {{
        ?technique a attack:Technique ; dcterm:title ?techniqueLabel .
        OPTIONAL {{ ?technique dcterm:description ?description }}
        BIND(REPLACE(STR(?technique), "^.*/", "") AS ?techniqueId)
        FILTER(REGEX(STR(?techniqueId), "^T[0-9]{{4}}(\\\\.[0-9]{{3}})?$"))
        FILTER({matches})
        OPTIONAL {{
            ?technique attack:hasMitigation ?mitigation .
            ?mitigation dcterm:title ?mitigationLabel .
        }}
    }}
    ORDER BY (IF({title_matches}, 0, 1))
             ?techniqueLabel ?mitigationLabel
    LIMIT 10
    """
    return execute_sparql_query(query, ctx)


@mcp.tool()
def get_technique_by_id(technique_id: str, ctx: Context, include_description: bool = True) -> str:
    """Get a specific MITRE ATT&CK technique or sub-technique by its ATT&CK ID.

    Use this when the question references a technique by ID (e.g. "What is
    T1505.003?") rather than by name -- get_techniques_by_keyword and the
    other technique tools only match against the technique's title/description
    text, not its ID, so an ID-only question returns no results there even
    though the technique exists.

    Args:
        technique_id: ATT&CK technique ID, e.g. 'T1505.003' or 'T1110'
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: True)
    """
    normalized_id = technique_id.strip().upper()
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>

    SELECT DISTINCT ?technique ?label ?description ?tacticLabel WHERE {{
        ?technique a attack:Technique .
        FILTER(STRENDS(STR(?technique), "/{normalized_id}"))
        ?technique dcterm:title ?label .
        OPTIONAL {{ ?technique dcterm:description ?description }}
        OPTIONAL {{
            ?technique attack:accomplishesTactic ?tactic .
            ?tactic dcterm:title ?tacticLabel .
        }}
    }}
    """
    return execute_sparql_query(query, ctx, include_description)


@mcp.tool()
def get_techniques_by_tactic(tactic_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all techniques that accomplish a specific tactic.
    
    Args:
        tactic_name: Name of the tactic to search for
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?techniqueLabel ?tactic ?tacticLabel WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?techniqueLabel .
        ?technique attack:accomplishesTactic ?tactic .
        ?tactic dcterm:title ?tacticLabel .
        FILTER(CONTAINS(LCASE(?tacticLabel), LCASE("{tactic_name}")))
    }}
    ORDER BY ?techniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_subtechniques_of_technique(technique_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all subtechniques of a parent technique.
    
    Args:
        technique_name: Name of the parent technique
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?subtechnique ?subtechniqueLabel ?parentTechnique ?parentLabel WHERE {{
        ?subtechnique a attack:SubTechnique .
        ?subtechnique dcterm:title ?subtechniqueLabel .
        ?subtechnique attack:isSubTechniqueOf ?parentTechnique .
        ?parentTechnique dcterm:title ?parentLabel .
        FILTER(CONTAINS(LCASE(?parentLabel), LCASE("{technique_name}")))
    }}
    ORDER BY ?subtechniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_techniques_by_platform(platform: str, ctx: Context, include_description: bool = False) -> str:
    """Get techniques that target a specific platform.
    
    Args:
        platform: Platform name (e.g., Windows, Linux, macOS)
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    # ?platform is bound to a URI resource (e.g. .../attack/platform/Linux),
    # not a string literal -- LCASE(?platform) directly errors on Virtuoso
    # ("SL001: LCASE() needs a string value as 1st argument"). STR() first
    # converts it to its string form (the platform name survives as the
    # URI's last path segment, so CONTAINS still matches correctly).
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>

    SELECT ?technique ?label ?platform WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?label .
        ?technique attack:platform ?platform .
        FILTER(CONTAINS(LCASE(STR(?platform)), LCASE("{platform}")))
    }}
    ORDER BY ?label
    """

    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Adversary Group Query Tools
#################################################################

@mcp.tool()
def get_all_adversary_groups(ctx: Context, include_description: bool = False) -> str:
    """Get all adversary groups in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?group ?label ?aliases WHERE {
        ?group a attack:AdversaryGroup .
        ?group dcterm:title ?label .
        OPTIONAL { ?group attack:aliases ?aliases }
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_techniques_used_by_group(group_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all techniques used by a specific adversary group.
    
    Args:
        group_name: Name of the adversary group
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?group ?groupLabel ?technique ?techniqueLabel WHERE {{
        ?group a attack:AdversaryGroup .
        ?group dcterm:title ?groupLabel .
        ?group attack:usesTechnique ?technique .
        ?technique dcterm:title ?techniqueLabel .
        FILTER(CONTAINS(LCASE(?groupLabel), LCASE("{group_name}")))
    }}
    ORDER BY ?techniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_software_used_by_group(group_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all software used by a specific adversary group.
    
    Args:
        group_name: Name of the adversary group
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?group ?groupLabel ?software ?softwareLabel WHERE {{
        ?group a attack:AdversaryGroup .
        ?group dcterm:title ?groupLabel .
        {{
            ?group attack:usesSoftware ?software .
        }} UNION {{
            ?group attack:usesMalware ?software .
        }}
        ?software dcterm:title ?softwareLabel .
        FILTER(CONTAINS(LCASE(?groupLabel), LCASE("{group_name}")))
    }}
    ORDER BY ?softwareLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_groups_using_technique(technique_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all adversary groups that use a specific technique.
    
    Args:
        technique_name: Name of the technique
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?group ?groupLabel ?technique ?techniqueLabel WHERE {{
        ?group a attack:AdversaryGroup .
        ?group dcterm:title ?groupLabel .
        ?group attack:usesTechnique ?technique .
        ?technique dcterm:title ?techniqueLabel .
        FILTER(CONTAINS(LCASE(?techniqueLabel), LCASE("{technique_name}")))
    }}
    ORDER BY ?groupLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Software and Malware Query Tools
#################################################################

@mcp.tool()
def get_all_software(ctx: Context, include_description: bool = False) -> str:
    """Get all software in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?software ?label ?type WHERE {
        ?software a ?type .
        ?software dcterm:title ?label .
        FILTER(?type = attack:Software || ?type = attack:Malware)
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_software_by_keyword(ctx: Context, keyword: str, include_description: bool = False) -> str:
    """Get all software in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?software ?label ?type WHERE {{
        ?software a ?type .
        ?software dcterm:title ?label .
        FILTER(?type = attack:Software || ?type = attack:Malware)
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()

def get_techniques_used_by_software(software_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all techniques implemented by specific software/malware.
    
    Args:
        software_name: Name of the software or malware
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?software ?softwareLabel ?technique ?techniqueLabel WHERE {{
        {{
            ?software a attack:Software .
            ?software dcterm:title ?softwareLabel .
            ?technique attack:hasSoftware ?software .
        }} UNION {{
            ?software a attack:Malware .
            ?software dcterm:title ?softwareLabel .
            ?software attack:implementsTechnique ?technique .
        }}
        ?technique dcterm:title ?techniqueLabel .
        FILTER(CONTAINS(LCASE(?softwareLabel), LCASE("{software_name}")))
    }}
    ORDER BY ?techniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Mitigation Query Tools
#################################################################

@mcp.tool()
def get_all_mitigations(ctx: Context, include_description: bool = False) -> str:
    """Get all mitigations in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?mitigation ?label WHERE {
        ?mitigation a attack:Mitigation .
        ?mitigation dcterm:title ?label .
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

def get_all_mitigations_by_keyword(ctx: Context, keyword: str, include_description: bool = False) -> str:
    """Get all mitigations in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?mitigation ?label WHERE {{
        ?mitigation a attack:Mitigation .
        ?mitigation dcterm:title ?label .
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_techniques_mitigated_by_mitigation(mitigation_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all techniques that are mitigated by a specific mitigation.
    
    Args:
        mitigation_name: Name of the mitigation
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?mitigation ?mitigationLabel ?technique ?techniqueLabel WHERE {{
        ?mitigation a attack:Mitigation .
        ?mitigation dcterm:title ?mitigationLabel .
        ?mitigation attack:preventsTechnique ?technique .
        ?technique dcterm:title ?techniqueLabel .
        FILTER(CONTAINS(LCASE(?mitigationLabel), LCASE("{mitigation_name}")))
    }}
    ORDER BY ?techniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_mitigations_for_technique(technique_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all mitigations that can prevent a specific technique.
    
    Args:
        technique_name: Name of the technique
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    # The underlying CSKG has duplicate RDF triples and parallel canonical
    # (T1078) / slug (valid-accounts) URIs for the same real technique or
    # mitigation -- selecting the raw ?technique/?mitigation URIs multiplies
    # those duplicates together in the join, producing dozens of rows that
    # are semantically identical. The URIs aren't informative to read anyway
    # (e.g. "course-of-action--f9f9e6ef-..."), so project + DISTINCT on just
    # the human-readable labels instead, which collapses both duplication
    # sources at once.
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>

    SELECT DISTINCT ?techniqueLabel ?mitigationLabel WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?techniqueLabel .
        ?technique attack:hasMitigation ?mitigation .
        ?mitigation dcterm:title ?mitigationLabel .
        FILTER(CONTAINS(LCASE(?techniqueLabel), LCASE("{technique_name}")))
    }}
    ORDER BY ?mitigationLabel
    """

    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Tactic Query Tools
#################################################################

@mcp.tool()
def get_all_tactics(ctx: Context, include_description: bool = False) -> str:
    """Get all tactics in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?tactic ?label WHERE {
        ?tactic a attack:Tactic .
        ?tactic dcterm:title ?label .
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_tactics_by_keyword(ctx: Context, keyword:str, include_description: bool = False) -> str:
    """Get all tactics in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?tactic ?label WHERE {{
        ?tactic a attack:Tactic .
        ?tactic dcterm:title ?label .
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_tactics_for_technique(technique_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all tactics accomplished by a specific technique.
    
    Args:
        technique_name: Name of the technique
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?techniqueLabel ?tactic ?tacticLabel WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?techniqueLabel .
        ?technique attack:accomplishesTactic ?tactic .
        ?tactic dcterm:title ?tacticLabel .
        FILTER(CONTAINS(LCASE(?techniqueLabel), LCASE("{technique_name}")))
    }}
    ORDER BY ?tacticLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Asset Query Tools (for ICS)
#################################################################

@mcp.tool()
def get_all_assets(ctx: Context, include_description: bool = False) -> str:
    """Get all assets in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?asset ?label WHERE {
        ?asset a attack:Asset .
        ?asset dcterm:title ?label .
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_assets_by_keyword(ctx: Context, keyword:str, include_description: bool = False) -> str:
    """Get all assets in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?asset ?label WHERE {{
        ?asset a attack:Asset .
        ?asset dcterm:title ?label .
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}")) ||
            CONTAINS(LCASE(?description), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_techniques_targeting_asset(asset_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all techniques that target a specific asset.
    
    Args:
        asset_name: Name of the asset
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?techniqueLabel ?asset ?assetLabel WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?techniqueLabel .
        ?technique attack:targetsAsset ?asset .
        ?asset dcterm:title ?assetLabel .
        FILTER(CONTAINS(LCASE(?assetLabel), LCASE("{asset_name}")))
    }}
    ORDER BY ?techniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Data Source and Component Query Tools
#################################################################

@mcp.tool()
async def get_all_data_sources(ctx: Context, include_description: bool = False) -> str:
    """Get all data sources in the MITRE ATT&CK framework.
    
    Args:
        include_description: Whether to include descriptions (default: False)
    """
     
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?dataSource ?label WHERE {
        ?dataSource a attack:DataSource .
        ?dataSource dcterm:title ?label .
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
async def get_data_sources_by_keyword(ctx: Context, keyword:str, include_description: bool = False) -> str:
    """Get all data sources in the MITRE ATT&CK framework.
    
    Args:
        include_description: Whether to include descriptions (default: False)
    """
     
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?dataSource ?label WHERE {{
        ?dataSource a attack:DataSource .
        ?dataSource dcterm:title ?label .
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
async def get_all_data_components(ctx: Context, include_description: bool = False) -> str:
    """Get all data components in the MITRE ATT&CK framework.
    
    Args:
        include_description: Whether to include descriptions (default: False)
    """    
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?dataComponent ?label WHERE {
        ?dataComponent a attack:DataComponent .
        ?dataComponent dcterm:title ?label .
    }
    ORDER BY ?label
    """
    

    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Complex Relationship Queries
#################################################################

@mcp.tool()
async def get_technique_relationships(technique_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get comprehensive relationships for a specific technique.
    
    Args:
        technique_name: Name of the technique
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?techniqueLabel ?relationshipType ?relatedEntity ?relatedLabel WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?techniqueLabel .
        FILTER(CONTAINS(LCASE(?techniqueLabel), LCASE("{technique_name}")))
        
        {{
            ?technique attack:accomplishesTactic ?relatedEntity .
            ?relatedEntity dcterm:title ?relatedLabel .
            BIND("accomplishes_tactic" AS ?relationshipType)
        }} UNION {{
            ?technique attack:hasMitigation ?relatedEntity .
            ?relatedEntity dcterm:title ?relatedLabel .
            BIND("has_mitigation" AS ?relationshipType)
        }} UNION {{
            ?technique attack:hasSoftware ?relatedEntity .
            ?relatedEntity dcterm:title ?relatedLabel .
            BIND("has_software" AS ?relationshipType)
        }} UNION {{
            ?relatedEntity attack:usesTechnique ?technique .
            ?relatedEntity dcterm:title ?relatedLabel .
            BIND("used_by_group" AS ?relationshipType)
        }} UNION {{
            ?technique attack:targetsAsset ?relatedEntity .
            ?relatedEntity dcterm:title ?relatedLabel .
            BIND("targets_asset" AS ?relationshipType)
        }}
    }}
    ORDER BY ?relationshipType ?relatedLabel
    """
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
async def get_group_capabilities(group_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get comprehensive capabilities (techniques, software, malware) for an adversary group.
    
    Args:
        group_name: Name of the adversary group
        include_description: Whether to include descriptions (default: False)
    """
   
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?group ?groupLabel ?capabilityType ?capability ?capabilityLabel WHERE {{
        ?group a attack:AdversaryGroup .
        ?group dcterm:title ?groupLabel .
        FILTER(CONTAINS(LCASE(?groupLabel), LCASE("{group_name}")))
        
        {{
            ?group attack:usesTechnique ?capability .
            ?capability dcterm:title ?capabilityLabel .
            BIND("technique" AS ?capabilityType)
        }} UNION {{
            ?group attack:usesSoftware ?capability .
            ?capability dcterm:title ?capabilityLabel .
            BIND("software" AS ?capabilityType)
        }} UNION {{
            ?group attack:usesMalware ?capability .
            ?capability dcterm:title ?capabilityLabel .
            BIND("malware" AS ?capabilityType)
        }}
    }}
    ORDER BY ?capabilityType ?capabilityLabel
    """
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# CVE Query Tools
#################################################################
# Written for the SEPSES-KG schema served at https://sepses.org/sparql
# (verified 2026-10-26): CVE IRIs are .../resource/cve/<CVE-ID> with
# dcterms:identifier/description/issued/modified and cve:hasReference;
# CVSS v3 records are NOT linked from the CVE but sit at the parallel IRI
# .../resource/cvss/CVSS3BaseMetric/<CVE-ID>, with a numeric cvss:baseScore,
# cvss:baseSeverity and cvss:vectorString. CVSS v2 is no longer present.

_CVE_PREFIXES = """
    PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
    PREFIX cvss: <http://w3id.org/sepses/vocab/ref/cvss#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""
_CVE_IRI = "http://w3id.org/sepses/resource/cve/"
_CVSS3_IRI = "http://w3id.org/sepses/resource/cvss/CVSS3BaseMetric/"
_CVE_ID_RE = re.compile(r"^CVE-\d{4}-\d{4,}$")
# From a CVSS v3 record to its CVE (cheap: filters on the score first).
_CVSS_TO_CVE = f'BIND(IRI(CONCAT("{_CVE_IRI}", STRAFTER(STR(?cvss3), "/CVSS3BaseMetric/"))) AS ?cve)'
# From a CVE to its (optional) CVSS v3 score and severity. The BIND must sit
# outside the OPTIONAL: inside it, ?cve is not yet bound and the join
# silently pairs every CVE with every CVSS record.
_CVE_TO_CVSS = (f'BIND(IRI(CONCAT("{_CVSS3_IRI}", STRAFTER(STR(?cve), "/cve/"))) AS ?cvss3) '
                f'OPTIONAL {{ ?cvss3 cvss:baseScore ?baseScore . '
                f'OPTIONAL {{ ?cvss3 cvss:baseSeverity ?sev }} }}')


def _normalize_cve_id(cve_id: str) -> Optional[str]:
    normalized = (cve_id or "").strip().upper()
    return normalized if _CVE_ID_RE.match(normalized) else None


def _literal(text: str) -> str:
    """Escape user text for a SPARQL string literal."""
    return (text or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


@mcp.tool()
def get_all_cves(ctx: Context, include_description: bool = False) -> str:
    """Get a sample of CVEs in the knowledge base (first 50).

    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = _CVE_PREFIXES + """
    SELECT ?cveId ?description WHERE {
        ?cve a cve:CVE ; dcterms:identifier ?cveId .
        OPTIONAL { ?cve dcterms:description ?description }
    }
    LIMIT 50
    """
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_cve_by_id(cve_id: str, ctx: Context, include_description: bool = False) -> str:
    """Get detailed information about a specific CVE, including its CVSS v3 score.

    Args:
        cve_id: CVE identifier (e.g., CVE-2023-1234)
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    normalized = _normalize_cve_id(cve_id)
    if not normalized:
        return f"Error: {cve_id!r} is not a CVE identifier (expected e.g. CVE-2021-44228)."
    query = _CVE_PREFIXES + f"""
    SELECT ?cveId ?description ?publishedDate ?modifiedDate ?baseScore ?severity ?vector WHERE {{
        VALUES ?cve {{ <{_CVE_IRI}{normalized}> }}
        ?cve a cve:CVE .
        BIND("{normalized}" AS ?cveId)
        OPTIONAL {{ ?cve dcterms:description ?description }}
        OPTIONAL {{ ?cve dcterms:issued ?publishedDate }}
        OPTIONAL {{ ?cve dcterms:modified ?modifiedDate }}
        OPTIONAL {{
            <{_CVSS3_IRI}{normalized}> cvss:baseScore ?baseScore .
            OPTIONAL {{ <{_CVSS3_IRI}{normalized}> cvss:vectorString ?vector }}
            OPTIONAL {{ <{_CVSS3_IRI}{normalized}> cvss:baseSeverity ?sev }}
        }}
        BIND(REPLACE(STR(?sev), "^.*/", "") AS ?severity)
    }}
    """
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def search_cves_by_keyword(keyword: str, ctx: Context, include_description: bool = False) -> str:
    """Search CVEs by keyword in their description (first 50 matches).

    Args:
        keyword: Keyword to search for
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = _CVE_PREFIXES + f"""
    SELECT ?cveId ?description WHERE {{
        ?cve a cve:CVE ; dcterms:identifier ?cveId ; dcterms:description ?description .
        FILTER(CONTAINS(LCASE(?description), LCASE("{_literal(keyword)}")))
    }}
    LIMIT 50
    """
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# CVSS Query Tools
#################################################################

def _cves_by_score(ctx: Context, minimum: float, maximum: float, include_description: bool) -> str:
    query = _CVE_PREFIXES + f"""
    SELECT ?cveId ?baseScore ?severity ?description WHERE {{
        ?cvss3 a cvss:CVSS3BaseMetric ; cvss:baseScore ?baseScore .
        FILTER(xsd:decimal(?baseScore) >= {float(minimum)} && xsd:decimal(?baseScore) <= {float(maximum)})
        OPTIONAL {{ ?cvss3 cvss:baseSeverity ?sev }}
        BIND(REPLACE(STR(?sev), "^.*/", "") AS ?severity)
        BIND(STRAFTER(STR(?cvss3), "/CVSS3BaseMetric/") AS ?cveId)
        {_CVSS_TO_CVE}
        OPTIONAL {{ ?cve dcterms:description ?description }}
    }}
    ORDER BY DESC(?baseScore)
    LIMIT 50
    """
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_cves_by_cvss_score(min_score: float, max_score: float, ctx:Context, include_description: bool = False) -> str:
    """Get CVEs within a specific CVSS v3 base-score range (highest first, max 50).

    Args:
        min_score: Minimum CVSS score
        max_score: Maximum CVSS score
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    return _cves_by_score(ctx, min_score, max_score, include_description)

@mcp.tool()
def get_high_severity_cves(ctx: Context, include_description: bool = False) -> str:
    """Get CVEs with high severity (CVSS v3 score >= 7.0; highest first, max 50).

    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    return _cves_by_score(ctx, 7.0, 10.0, include_description)

@mcp.tool()
def get_critical_cves(ctx: Context, include_description: bool = False) -> str:
    """Get CVEs with critical severity (CVSS v3 score >= 9.0; highest first, max 50).

    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    return _cves_by_score(ctx, 9.0, 10.0, include_description)

#################################################################
# Reference Query Tools
#################################################################

@mcp.tool()
def get_references_for_cve(cve_id: str, ctx: Context, include_description: bool = False) -> str:
    """Get the reference URLs (advisories, patches, exploits) for a specific CVE.

    Args:
        cve_id: CVE identifier (e.g., CVE-2023-1234)
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    normalized = _normalize_cve_id(cve_id)
    if not normalized:
        return f"Error: {cve_id!r} is not a CVE identifier (expected e.g. CVE-2021-44228)."
    # One reference node per CVE holds all URLs, sources and tag lists as
    # separate values, so tags cannot be paired with individual URLs.
    query = _CVE_PREFIXES + f"""
    SELECT DISTINCT ?referenceUrl ?referenceSource WHERE {{
        <{_CVE_IRI}{normalized}> cve:hasReference ?reference .
        ?reference cve:referenceUrl ?referenceUrl .
        OPTIONAL {{ ?reference cve:referenceSource ?referenceSource }}
    }}
    ORDER BY ?referenceUrl
    """
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Time-based Query Tools
#################################################################

def _cves_published_between(ctx: Context, start: str, end: Optional[str], include_description: bool) -> str:
    upper = f' && ?publishedDate < "{end}"^^xsd:dateTime' if end else ""
    query = _CVE_PREFIXES + f"""
    SELECT ?cveId ?publishedDate ?baseScore ?severity ?description WHERE {{
        ?cve a cve:CVE ; dcterms:issued ?publishedDate ; dcterms:identifier ?cveId .
        FILTER(?publishedDate >= "{start}"^^xsd:dateTime{upper})
        OPTIONAL {{ ?cve dcterms:description ?description }}
        {_CVE_TO_CVSS}
        BIND(REPLACE(STR(?sev), "^.*/", "") AS ?severity)
    }}
    ORDER BY DESC(?publishedDate)
    LIMIT 100
    """
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_recent_cves(ctx: Context, days: int = 30, include_description: bool = False) -> str:
    """Get CVEs published in the last N days (newest first, max 100).

    Args:
        ctx: FastMCP context object
        days: Number of days to look back (default: 30)
        include_description: Whether to include descriptions (default: False)
    """
    from datetime import datetime, timedelta, timezone
    days = max(1, int(days))
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    return _cves_published_between(ctx, start, None, include_description)

@mcp.tool()
def get_cves_by_year(year: int, ctx: Context, include_description: bool = False) -> str:
    """Get CVEs published in a specific year (newest first, max 100).

    Args:
        year: Year to filter by (e.g., 2023)
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    year = int(year)
    return _cves_published_between(ctx, f"{year}-01-01T00:00:00", f"{year + 1}-01-01T00:00:00", include_description)

#################################################################
# Free-form fallback: text -> SPARQL
#################################################################
# The ~40 tools above each answer one fixed, narrowly-scoped question
# shape. This is the fallback for anything else the schema *can* answer
# but no fixed tool covers -- schema-grounded LLM query generation, the
# same approach src/agents/cypher_agent.py uses for Neo4j, adapted to this
# server's RDF/SPARQL schema. (This replaces a previous implementation
# that was broken two ways at once: registered with @mcp.prompt() instead
# of @mcp.tool() -- a different MCP primitive an autonomous tool-calling
# agent never sees in its tool list at all -- AND defined *after* this
# module's `if __name__ == "__main__": mcp.run()` entry point, so even its
# decorator never ran in the actual server process; it was never
# registered as anything, let alone reachable. Its body was also a
# hardcoded placeholder that ignored the input prompt entirely.)

SPARQL_SCHEMA_DESCRIPTION = """
Namespaces (always declare the ones you use as PREFIX lines):
  attack: <http://w3id.org/sepses/vocab/ref/attack#>
  cve:    <http://w3id.org/sepses/vocab/ref/cve#>
  cvss:   <http://w3id.org/sepses/vocab/ref/cvss#>
  capec:  <http://w3id.org/sepses/vocab/ref/capec#>
  dcterm: <http://purl.org/dc/terms/>

Classes: attack:Technique, attack:SubTechnique, attack:Tactic,
  attack:AdversaryGroup, attack:Software, attack:Malware, attack:Mitigation,
  cve:CVE

Common triple patterns actually used by this knowledge base's own tools --
follow these shapes exactly, don't invent alternate predicate names:
  ?technique a attack:Technique ; dcterm:title ?label ;
    dcterm:description ?desc ; attack:accomplishesTactic ?tactic .
  ?subtechnique a attack:SubTechnique ; attack:isSubTechniqueOf ?parentTechnique .
  ?tactic a attack:Tactic ; dcterm:title ?label .
  ?group a attack:AdversaryGroup ; dcterm:title ?label ; attack:aliases ?aliases ;
    attack:usesTechnique ?technique ; attack:usesSoftware ?software ;
    attack:usesMalware ?malware .
  ?mitigation a attack:Mitigation ; dcterm:title ?label ;
    attack:preventsTechnique ?technique .
    (equivalently, from the technique side: ?technique attack:hasMitigation ?mitigation)
  ?software a attack:Software|attack:Malware ; dcterm:title ?label .
  ?cve a cve:CVE ; dcterms:identifier ?cveId ; dcterms:description ?desc ;
    dcterms:issued ?published ; dcterms:modified ?modified .
  CVSS v3 is NOT linked from the CVE: it sits at the parallel IRI
    <http://w3id.org/sepses/resource/cvss/CVSS3BaseMetric/CVE-2021-44228>
    with cvss:baseScore (numeric) and cvss:baseSeverity. Join with
    BIND(IRI(CONCAT("http://w3id.org/sepses/resource/cvss/CVSS3BaseMetric/",
    STRAFTER(STR(?cve), "/cve/"))) AS ?cvss3).

Rules, matching this knowledge base's own conventions:
  - Match technique/tactic/group/mitigation/software NAMES via
    FILTER(CONTAINS(LCASE(?label), LCASE("..."))) on dcterm:title -- there is
    no exact-match property.
  - Match a technique by its ATT&CK ID (e.g. "T1110") via
    FILTER(STRENDS(STR(?technique), "/T1110")) -- the ID is the URI's last
    path segment, not a literal property value.
  - Compare ?baseScore numerically with xsd:decimal(?baseScore)
    (declare PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>).
  - Never mix ATT&CK and CVE patterns in one query: the two live in
    different stores (ATT&CK local file, CVE remote endpoint).
  - Always SELECT human-readable ?label/?title variables alongside any URI
    variable -- raw URIs are not useful to a downstream reader.
  - This knowledge base has NO IP addresses, hostnames, domains, live
    threat-intel feeds, or organization-specific data of any kind -- only
    public MITRE ATT&CK (techniques/tactics/groups/software/mitigations)
    and CVE/CVSS records. If the question needs any of that, it cannot be
    answered from this schema at all.
"""


def _generate_sparql(question: str) -> Optional[str]:
    """One-shot LLM call translating a natural-language question into a
    single read-only SPARQL SELECT query grounded in the schema above.
    Returns None if the model determines the question is outside what
    this knowledge base can answer at all (e.g. IP reputation, live
    threat feeds) -- forcing a query in that case would just waste a
    round-trip returning nothing useful -- or if generation/the API key
    itself fails."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("[text_to_sparql] OPENAI_API_KEY not set -- cannot generate a query")
        return None
    import uuid
    call_id = uuid.uuid4().hex
    _eval_support.event('llm_started', call_id=call_id, model='gpt-4o', provider='openai')
    try:
        client = OpenAI(api_key=api_key, **({'max_retries': 0, 'timeout': 45} if os.environ.get('PROPOSAL_TELEMETRY_PATH') else {}))
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You translate a natural-language question into a single read-only "
                        "SPARQL SELECT query against the knowledge graph schema below. Output "
                        "ONLY the raw SPARQL query text -- no markdown code fences, no "
                        "explanation. If the question cannot be answered from this schema at "
                        "all, output exactly the single word: NO_QUERY\n\n" + SPARQL_SCHEMA_DESCRIPTION
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        _eval_support.event('llm_completed', call_id=call_id, model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None)
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:
        _eval_support.event('llm_failed', call_id=call_id, error_type=type(e).__name__)
        logger.error(f"[text_to_sparql] SPARQL generation call failed: {e}")
        return None

    if raw.upper().startswith("NO_QUERY"):
        logger.info(f"[text_to_sparql] model determined this schema can't answer: {question!r}")
        return None

    # Defensive: strip markdown fences if the model adds them despite being
    # told not to (observed behavior from structured-output-averse prompts
    # elsewhere in this codebase's history).
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("sparql"):
            raw = raw[len("sparql"):]
    query = raw.strip()

    # Same defense-in-depth posture as src/config/settings.py::
    # ReadOnlyNeo4jGraph earlier in this project: reject any SPARQL Update
    # keyword before the query ever reaches the endpoint. The public SEPSES
    # endpoint only exposes a query (not update) protocol surface anyway,
    # but a locally-configured --rdf-file/local-graph mode has no such
    # protocol-level restriction, so this check is the only guard in that
    # mode.
    forbidden = re.compile(r"\b(INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|COPY|MOVE|ADD)\b", re.IGNORECASE)
    if forbidden.search(query):
        logger.error(f"[text_to_sparql] generated query rejected (write keyword present): {query}")
        return None
    return query


@mcp.tool()
def text_to_sparql(prompt: str, ctx: Context, include_description: bool = False) -> str:
    """Answer an open-ended question none of the other ~40 fixed tools
    directly cover, by generating and executing a SPARQL query against the
    MITRE ATT&CK / CVE / CVSS schema this server exposes. Check whether a
    more specific tool already answers the question directly first (e.g.
    get_technique_by_id for an ID lookup, get_cves_by_cvss_score for a
    score range) -- those are cheaper and more reliable than a generated
    query. This tool cannot answer questions needing data outside this
    schema (IP reputation, live threat feeds, organization-specific data)
    -- it will say so plainly rather than guessing.

    Args:
        prompt: The natural-language question to answer.
        ctx: FastMCP context object.
        include_description: Whether to include full description text in results (default: False).
    """
    query = _generate_sparql(prompt)
    if query is None:
        return (
            "This question cannot be answered from this knowledge base -- it only covers "
            "public MITRE ATT&CK (techniques, tactics, adversary groups, software, "
            "mitigations) and CVE/CVSS data. It has no IP addresses, hostnames, live threat "
            "feeds, or organization-specific data."
        )
    logger.info(f"[text_to_sparql] prompt={prompt!r} -> generated query:\n{query}")
    return execute_sparql_query(query, ctx, include_description)


# Run the server
if __name__ == "__main__":
    logger.info("Starting mcp.run()")
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start RDF Explorer: {str(e)}")
        sys.exit(1)
    logger.info("mcp.run() completed")
