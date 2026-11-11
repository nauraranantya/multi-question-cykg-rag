# src/agents/cypher_agent.py
import re
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from src.config.settings import read_only_graph, llm, fast_llm
from src.retrieval.temporal import weight_and_sort_records
from src.retrieval import case_scope as cs

# --- Cypher Generation Prompt Template ---
# Schema + examples below match src/ingestion/graph_loader.py exactly (the
# actual graph-construction code in this repo, not the earlier no-code
# hosted tool the original AgCyRAG paper assumed -- see
# ../../defengraph_vs_agcyrag_analysis.md Sec.3.2). Earlier examples in this
# prompt referenced FAILED_LOGIN/SESSION_OPENED_ON/AUTHENTICATION_FAILURE_ON
# relationship types that don't exist anywhere in graph_loader.py -- a stale
# holdover from a different demo schema. Every relationship type below is
# one graph_loader.py actually writes.
cypher_generation_template = """
You are an expert Neo4j Cypher translator investigating security alerts, converting an
investigative question into Cypher based on the Neo4j Schema provided.

Before writing the query, first work out what data the question actually needs: (a) which entity
it's anchored on (a specific Host/IP/User/Alert), and (b) whether it needs a KNOWN, SPECIFIC filter
value (a literal ID, technique ID, or timestamp) or whether it needs the alert's own descriptive
content (title, full_log, rule_level) returned so that content itself can answer the question.
Retrieve BROADLY around the anchor entity and return descriptive fields for the latter case --
never invent a speculative WHERE-clause filter using a guessed keyword (e.g. filtering for the
word "scanning" or "probing" in some property) just because it sounds like it should match the
hypothesis. A query that returns more rows than strictly needed (for downstream review to read and
interpret) is far better than one that returns nothing because a guessed filter value doesn't
actually match what's stored -- Suricata/Wazuh alert titles and full_log text already describe
what was detected in human-readable form; that description IS the evidence, not a keyword filter
invented to pre-judge it.

Following the instructions below:
        1. Generate Cypher query compatible ONLY for Neo4j Version 5.
        2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use an alias when using the WITH keyword.
        3. Use only Node labels and Relationship types mentioned in the schema.
        4. Do not use relationships that are not mentioned in the given schema.
        5. For property searches, use case-insensitive matching. E.g., to search for a User, use `toLower(u.id) CONTAINS 'search_term'`.
        6. Assign a meaningful alias to every node and relationship in the MATCH clause (e.g., `MATCH (h:Host)-[r:HAS_ALERT]->(a:Alert)`).
        7. In the RETURN clause, include only the components (nodes, relationships, or properties) needed to answer the question.
        8. To count distinct items from an `OPTIONAL MATCH`, collect them first and then use `size()` on the list to avoid null value warnings (e.g., `WITH main, collect(DISTINCT opt) AS items RETURN size(items) AS itemCount`).
        9. To create unique pairs of nodes for comparison, use `WHERE elementId(node1) < elementId(node2)`.
        10. **CRITICAL RULE**: When returning the `type()` of a relationship, you MUST give the relationship a variable in the `MATCH` clause. E.g., `MATCH (h:Host)-[r:HAS_ALERT]->(a:Alert) RETURN type(r)`. Do NOT use `type()` on a relationship without a variable.
        11. **CRITICAL RULE**: A `WITH`/`RETURN` clause using `DISTINCT` or an aggregate function (`count()`, `collect()`, etc.) drops every variable not explicitly named in that clause -- a later `ORDER BY` can ONLY reference variables that clause actually returned, never an earlier MATCH variable that fell out of scope. E.g. `MATCH (h:Host)-[:HAS_ALERT]->(a:Alert) RETURN DISTINCT h.id AS hostId ORDER BY a.timestamp` is INVALID (`a` is gone after the DISTINCT); either return `a.timestamp` as an alias and order by that alias, or drop DISTINCT/the aggregate if you need to keep ordering by an unreturned variable.
        12. **CRITICAL RULE**: `MitreTechnique.tactic` is populated ONLY from the original alert's own native tag (most alerts don't have one), and when present it is always one of the ~14 standard MITRE ATT&CK tactic names verbatim (e.g. "Discovery", "Lateral Movement", "Initial Access", "Command and Control") -- it is NEVER a hypothesis's own free-text wording. Words like "scanning", "probing", or "reconnaissance" will NEVER appear in `m.tactic` even for genuinely related alerts, so `WHERE toLower(m.tactic) CONTAINS "scanning"` is guaranteed to return nothing regardless of whether relevant data exists. If the question names a SPECIFIC known technique, filter by `m.id` (e.g. `m.id = "T1046"` or `m.id STARTS WITH "T1046"`) -- that is the reliable, structured identifier. If the question is more open-ended (e.g. "does this look like scanning activity"), do NOT filter MitreTechnique by any guessed tactic text at all -- retrieve the alert's own `title`/`full_log` instead (per the guidance above) and let that content answer it.
        13. **CRITICAL RULE**: `Alert.timestamp` is a plain STRING property (ISO-8601), not a native Cypher temporal type. Comparing it directly against another ISO-8601 string works fine (e.g. `a.timestamp >= "2022-01-24T00:00:00Z"`), but a relative time window (e.g. "in the last 24 hours") requires converting it first with `datetime(a.timestamp)`. Cypher's `duration` values do NOT have a `.toMillis()` method -- that is a Java/JavaScript API, not Cypher -- so `duration('P1D').toMillis()` is invalid and will fail with a syntax error. Subtract a `duration` directly from a `datetime` instead, e.g. `datetime(a.timestamp) >= datetime() - duration({{hours: 24}})`. Also, this data is historical, not near real wall-clock time: if a "Reference time" is given below (see the question), anchor relative-time phrases like "the last N hours" to THAT timestamp instead of `datetime()` with no arguments -- real wall-clock `datetime()` will never match historical data.

        14. **CRITICAL RULE**: Avoid `UNION`/`UNION ALL`. Combine alternatives with `OPTIONAL MATCH`, `collect()`, list comprehensions or `CALL {{ ... }}` subqueries instead. If you do use UNION, every branch MUST return exactly the same column names, in the same order (use `AS` aliases, and `null AS x` for a column a branch lacks); otherwise Neo4j rejects the whole query.
        15. **CRITICAL RULE**: IP, User and MitreTechnique edges exist only when the sensor parsed that field. Many alerts (for example anomaly-detector and log-file alerts) have NO `ATTACK_TO`/`CONNECTS_TO`/`TARGETS_USER` edge even when the IP or user appears in their `full_log`. A pattern such as `(ip:IP {{id: "x"}})-[:ATTACK_TO]->(a:Alert)` therefore silently misses them. When an IP, user or domain is the anchor, match the alert through `(h:Host)-[:HAS_ALERT]->(a:Alert)` and use `OPTIONAL MATCH` for the IP/User edge, OR'd with `a.full_log CONTAINS "x"`, e.g. `MATCH (h:Host)-[:HAS_ALERT]->(a:Alert) OPTIONAL MATCH (src:IP)-[:ATTACK_TO]->(a) WITH h, a, src WHERE src.id = "x" OR a.full_log CONTAINS "x"`.
        16. **CRITICAL RULE**: Return the evidence rows themselves (alert id, timestamp, host, title, full_log), not a verdict you computed in Cypher. Do not return derived booleans or labels such as `isScanThenExploit`, `wasSuccessful` or `firstSuccessTime` built from keyword filters: a keyword filter that misses the stored wording turns into a false "no". For "did X happen / in what order" questions, return the matching rows ordered by `a.timestamp`; for large volumes, return counts plus first/last timestamps per group (e.g. per host and title) with one example alert id each, so the answer still fits in a few rows.
        17. Keep the query to one statement, without comments, and use only double-quoted string literals.

Schema:
{schema}

Domain notes (this graph represents security alerts, not general logs):
- (Host)-[:HAS_ALERT]->(Alert): which host an alert fired on.
- (IP)-[:ATTACK_TO]->(Alert): the source/attacking IP for an alert, if any.
- (Alert)-[:CONNECTS_TO]->(IP): the destination IP for an alert, if any.
- (Alert)-[:TARGETS_USER]->(User): the targeted user account, if any.
- (Alert)-[:TRIGGERS]->(MitreTechnique {{id, tactic}}): MITRE ATT&CK technique(s) the alert was
  natively tagged with (not every alert has one -- absence is itself informative).
- Alert nodes carry: id, title, rule_id, rule_level (Wazuh severity, 0-15), timestamp (ISO-8601),
  full_log, scenario_id.
- Investigative questions about a security alert are usually one of: scope/correlation (what else
  is connected to this host/IP/user), temporal (what happened before/after, in what order),
  historical recurrence (has this host/IP/user triggered alerts before), or technique history
  (what MITRE techniques has this host/IP triggered). Prefer queries that traverse from the
  entity in question (Host/IP/User) outward, ordered by `a.timestamp`, over single-alert lookups.

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than for you to construct a Cypher statement.
Do not run any queries that would add to or delete from the database.

Examples:

1.  Question: What other alerts has this host triggered, in order?
    Query:
    MATCH (h:Host {{id: "webserver01"}})-[:HAS_ALERT]->(a:Alert)
    RETURN a.id AS alertId, a.title AS title, a.rule_level AS severity, a.timestamp AS timestamp
    ORDER BY a.timestamp

2.  Question: Has this source IP attacked any other hosts, and how many?
    Query:
    MATCH (ip:IP {{id: "203.0.113.7"}})-[:ATTACK_TO]->(a:Alert)<-[:HAS_ALERT]-(h:Host)
    RETURN count(DISTINCT h) AS hostsAffected, collect(DISTINCT h.id) AS hostIds

3.  Question: What MITRE ATT&CK techniques has this host triggered previously?
    Query:
    MATCH (h:Host {{id: "webserver01"}})-[:HAS_ALERT]->(a:Alert)-[:TRIGGERS]->(m:MitreTechnique)
    RETURN DISTINCT m.id AS techniqueId, m.tactic AS tactic, a.timestamp AS whenTriggered
    ORDER BY a.timestamp

4.  Question: Has this user been targeted by an alert before this timestamp?
    Query:
    MATCH (u:User {{id: "daryl"}})<-[:TARGETS_USER]-(a:Alert)
    WHERE a.timestamp < "2026-08-04T01:02:03Z"
    RETURN a.id AS alertId, a.title AS title, a.timestamp AS timestamp
    ORDER BY a.timestamp DESC
    LIMIT 5

5.  Question: Give me all activity connected to the alerts on this host.
    Query:
    MATCH (h:Host {{id: "webserver01"}})-[:HAS_ALERT]->(a:Alert)
    OPTIONAL MATCH (srcIp:IP)-[:ATTACK_TO]->(a)
    OPTIONAL MATCH (a)-[:CONNECTS_TO]->(dstIp:IP)
    OPTIONAL MATCH (a)-[:TARGETS_USER]->(u:User)
    OPTIONAL MATCH (a)-[:TRIGGERS]->(m:MitreTechnique)
    RETURN a.id AS alertId, a.timestamp AS timestamp, a.title AS title, a.rule_level AS severity, h.id AS host, srcIp.id AS sourceIp,
           dstIp.id AS destIp, u.id AS targetUser, collect(DISTINCT m.id) AS techniques

6.  Question: Is there evidence of outbound scanning or probing activity from this IP?
    Query (WRONG way -- do not do this: `WHERE toLower(m.tactic) CONTAINS "scanning"` guesses at
    free-text that is never actually stored, guaranteeing an empty result either way):
    MATCH (ip:IP {{id: "10.143.2.4"}})-[:ATTACK_TO]->(a:Alert)
    WHERE toLower(a.title) CONTAINS "scan" OR toLower(a.full_log) CONTAINS "scan"
    RETURN a.id AS alertId, a.title AS title, a.rule_level AS severity, a.timestamp AS timestamp,
           a.full_log AS log
    ORDER BY a.timestamp
    (Right approach: retrieve every alert this IP triggered -- with its title/full_log, which
    already describes what was actually detected -- rather than pre-filtering on a guessed keyword
    that might not appear verbatim anywhere. If the volume is large, a light keyword filter on
    `a.title`/`a.full_log` themselves, as above, is fine since those fields do contain real
    descriptive text -- the mistake is filtering `m.tactic` on words that field never contains.)

7.  Question: How many alerts has this host triggered in the last 24 hours, and what were their
    severity levels? (Reference time for relative time phrases: "2022-01-25T23:58:41Z")
    Query (WRONG way -- do not do this: `duration('P1D').toMillis()` is not valid Cypher (Duration
    has no `.toMillis()` method), and real wall-clock `datetime()` would never match this historical
    data anyway):
    MATCH (h:Host {{id: "mail"}})-[:HAS_ALERT]->(a:Alert)
    WHERE a.timestamp >= datetime().epochMillis - duration('P1D').toMillis()
    RETURN count(a) AS alertCount
    (Right approach: convert the stored string with `datetime(a.timestamp)`, and anchor "the last 24
    hours" to the given reference time, not real wall-clock `datetime()`):
    MATCH (h:Host {{id: "mail"}})-[:HAS_ALERT]->(a:Alert)
    WHERE datetime(a.timestamp) >= datetime("2022-01-25T23:58:41Z") - duration({{hours: 24}})
    RETURN a.id AS alertId, a.rule_level AS severity, a.timestamp AS timestamp
    ORDER BY a.timestamp


The question is:
{question}
"""

cyper_generation_prompt = PromptTemplate(
    template=cypher_generation_template,
    input_variables=["schema","question"]
)

# --- Cypher QA Prompt Template ---
qa_template = """
You are a security analyst assistant that takes the results from a Neo4j security-alert graph query and forms a human-readable response. The query results section contains the results of a Cypher query that was generated based on an investigative question about a security alert. The provided information is authoritative; you must never question it or use your internal knowledge to alter it. Make the answer sound like a response to the question.
Final answer should be easily readable and structured.
Query Results:
{context}

Question: {question}
If the provided information is empty, respond by stating that you don't know the answer. Empty information is indicated by: []
If the information is not empty, you must provide an answer using the results. If the question involves a time duration, assume the query results are in units of days unless specified otherwise.
Never state that you lack sufficient information if data is present in the query results. Always utilize the data provided.
Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"]
)

# --- Cypher QA Chain and Query Function ---
# Built lazily (not at import time): GraphCypherQAChain.from_llm() eagerly
# calls graph.get_structured_schema, which -- now that `graph` is a lazy
# Neo4j proxy (see src/config/settings.py) -- would otherwise force a live
# Neo4j connection the instant this module is imported, even by code that
# never calls query_cypher() at all.
_cypher_qa_chain = None


def _get_cypher_qa_chain() -> GraphCypherQAChain:
    global _cypher_qa_chain
    if _cypher_qa_chain is None:
        _cypher_qa_chain = GraphCypherQAChain.from_llm(
            top_k=10,
            # A dedicated read-only connection (src/config/settings.py's
            # ReadOnlyNeo4jGraph), not the shared `graph` -- this is the
            # one place an LLM's own generated Cypher gets executed
            # against the live graph; see SECURITY_ASSESSMENT.md.
            graph=read_only_graph.resolve(),
            verbose=True,
            validate_cypher=True,
            return_intermediate_steps=True,
            cypher_prompt=cyper_generation_prompt,
            qa_prompt=qa_generation_prompt,
            qa_llm=fast_llm,
            cypher_llm=llm,
            allow_dangerous_requests=True,
            use_function_response=True
        )
    return _cypher_qa_chain


# Appended to the question in case-scoped mode. The scope itself is NOT
# left to the LLM -- src/retrieval/case_scope.py rewrites every Alert node
# pattern deterministically -- this only asks for the two columns that make
# each returned row attributable to an alert, and for no LIMIT of its own.
_SCOPE_NOTE = (
    "(Scope: this query is executed twice automatically -- once restricted to the member alerts of "
    "the case under investigation, once over all other alerts. Do NOT filter by a list of alert IDs "
    "yourself and do NOT add a LIMIT. Whenever a row describes an individual alert, return "
    "`a.id AS alertId` and `a.timestamp AS timestamp` among its columns.)"
)
# Identifiers that exist only in this module's prompt examples or are
# obvious stand-ins. In the first live v3 run two of five generated queries
# anchored on `{id: "alert-123"}` and `{id: "host-id"}` -- the question
# said "this alert"/"this host" and the generator, which never sees the
# case, filled the gap from the examples. Such a query returns nothing by
# construction, and "nothing" was then read as evidence. It is now a
# generation failure, recorded like any other failed retrieval.
_PLACEHOLDER_VALUE = (
    r"(alert-123|webserver01|203\.0\.113\.7|"
    r"(?:host|alert|user|ip)[-_ ]?(?:id|name|address)|<[^\"'>]*>|your[-_ ][^\"']*|example[^\"']*|REPLACE[^\"']*)")
# Matches the placeholder wherever an id is compared: `{id: "x"}`,
# `a.id = "x"`, `toLower(h.id) CONTAINS 'x'`, `a.id IN ["x"]`. The v3 run
# only needed the map form; the generator is free to write any of them.
_PLACEHOLDER_ID_RE = re.compile(
    r"\bid\)?\s*(?::|=|CONTAINS|STARTS\s+WITH|IN\s*\[)\s*[\"']" + _PLACEHOLDER_VALUE + r"[\"']",
    re.IGNORECASE)

# One regeneration, with the failure shown to the generator. The dispatcher's
# own retry repeats the same prompt, so a deterministic generation fault
# (placeholder anchor, rejected syntax) failed identically twice.
_GENERATION_ATTEMPTS = 3

# Filters on free text the generator cannot know: a tag that most alerts
# lack, or a literal it guessed. Only consulted when a query returns nothing.
_GUESSED_FILTER_RE = re.compile(
    r"\b\w+\.tactic\b|\b\w+\.(?:title|full_log)\)?\s*(?:CONTAINS|STARTS\s+WITH|=~?)\s*[\"']",
    re.IGNORECASE)


def _find_placeholder(query: str):
    match = _PLACEHOLDER_ID_RE.search(query or "")
    return match.group(1) if match else None


def _guessed_filters(query: str) -> list:
    return sorted({m.group(0).strip() for m in _GUESSED_FILTER_RE.finditer(query or "")})


def _repairable(exc: BaseException) -> bool:
    """A faulty query (placeholder, rejected schema, Neo4j statement error)
    can be rewritten; a timeout or lost connection cannot."""
    from src.retrieval import scoped_fetch
    return isinstance(exc, ValueError) or scoped_fetch.is_cypher_syntax_error(exc)


def _repair_feedback(previous: str, reason: str) -> str:
    return ("(Your previous query was rejected -- do not repeat it.\n"
            f"Previous query:\n{(previous or '').strip()[:1500]}\n"
            f"Problem: {reason[:600]}\n"
            "Write a corrected query. Quote string literals with double quotes and write no backticks inside them; "
            "anchor on the case's real host ID(s) or on no id at all; never use an identifier from the examples.)")


def _empty_feedback(previous: str, filters: list) -> str:
    return ("(Your previous query returned NO rows, and it filtered on text you could not have known "
            f"({'; '.join(filters)[:300]}). Previous query:\n{(previous or '').strip()[:1500]}\n"
            "Write a broader query: drop the guessed filter, keep the anchor entity, and return the alert's "
            "own `title`, `rule_level`, `full_log`, `timestamp` so its content can be read.)")


def _is_empty(context: dict) -> bool:
    try:
        return (context["case_evidence"]["total_rows"] == 0
                and context["historical_evidence"]["total_rows"] == 0)
    except (KeyError, TypeError):
        return False


def _entity_hint(case_scope: dict) -> str:
    """What the generator needs in order not to invent an anchor."""
    hosts = list(case_scope.get("hosts") or [])[:20]
    return (
        "(Entities: if the question says 'this alert', 'this case' or 'these alerts' without naming an "
        "alert ID, do not filter Alert nodes by id at all -- the scope restriction already selects the "
        "case's alerts. Never invent an identifier or reuse one from the examples."
        + (f" The case's host ID(s): {', '.join(hosts)}." if hosts else "") + ")")


# Rows pulled for the post-filter fallback (no DB-side scope available).
_POST_FILTER_FETCH_CAP = cs.CASE_FETCH_CAP + cs.HISTORICAL_FETCH_CAP


def _generate_cypher(question: str) -> str:
    """Same prompt, model, extraction and schema-based correction as
    GraphCypherQAChain._call, without its execute-then-slice step."""
    from langchain_neo4j.chains.graph_qa.cypher import extract_cypher
    from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
    from src.ingestion.graph_schema import SCHEMA_TEXT, RELATIONSHIPS
    chain = _get_cypher_qa_chain()
    generated = chain.cypher_generation_chain.invoke({"question": question, "schema": SCHEMA_TEXT})
    generated = extract_cypher(generated)
    # A ```cypher fence leaves its language tag as the first line. Neo4j
    # happens to accept a bare leading CYPHER keyword, but it is not part
    # of the query.
    generated = re.sub(r"^\s*cypher\s*\n", "", generated, flags=re.IGNORECASE)
    if not (generated or '').strip():
        raise ValueError('Cypher model returned an empty query before schema validation')
    # The observed schema can be sparse at startup and is cached by the chain.
    # A missing edge in that snapshot does not make its declared type invalid.
    corrected = CypherQueryCorrector([Schema(*r) for r in RELATIONSHIPS])(generated)
    if not corrected:
        raise ValueError('Cypher rejected by declared ingestion schema; generated query: ' + generated[:3000])
    return corrected.strip()


def run_scoped_cypher(generated: str, case_scope: dict, *, fetch=None, resolve=None,
                      is_syntax_error=None) -> dict:
    """Executes one generated query as two scoped queries and assembles
    the result (see src/retrieval/case_scope.py). `fetch`/`resolve` are
    injectable so this is testable without Neo4j.

    Enforcement tiers, recorded in the result's `scope_enforcement`:
      db_predicate  the case filter is part of the query Neo4j runs, so it
                    applies before ORDER BY/LIMIT/aggregation (normal path);
      post_filter   the query could not be rewritten (or this server
                    rejected the rewritten syntax): it is run once without
                    its trailing LIMIT and rows are partitioned client-side
                    by resolving the alert IDs they contain. Still never
                    mixes channels, but is only complete up to the fetch
                    cap -- reported in `notes`, never silent."""
    from src.retrieval import scoped_fetch
    fetch = fetch or scoped_fetch.fetch_capped
    resolve = resolve or scoped_fetch.resolve_alerts
    is_syntax_error = is_syntax_error or scoped_fetch.is_cypher_syntax_error
    reference_time = case_scope.get("reference_time")
    params = {cs.CASE_PARAM: list(case_scope["alert_ids"])}
    if reference_time:
        params[cs.CUTOFF_PARAM] = reference_time
    case_ids = set(case_scope["alert_ids"])
    notes, queries = [], {"generated": generated}

    enforcement = "db_predicate"
    try:
        scoped = cs.scope_cypher(generated, has_cutoff=bool(reference_time))
        case_rows, case_truncated = fetch(scoped["case"], params, cs.CASE_FETCH_CAP)
        hist_rows, hist_truncated = fetch(scoped["historical"], params, cs.HISTORICAL_FETCH_CAP)
        queries.update(case=scoped["case"], historical=scoped["historical"])
    except Exception as exc:
        if not (isinstance(exc, cs.ScopeRewriteError) or is_syntax_error(exc)):
            raise
        enforcement = "post_filter"
        notes.append(f"Case scope could not be enforced inside the database ({type(exc).__name__}: {exc}); "
                     f"rows were partitioned after retrieval instead.")
        unlimited = cs.strip_trailing_limit(generated)
        queries["post_filter"] = unlimited
        rows, truncated = fetch(unlimited, {}, _POST_FILTER_FETCH_CAP)
        meta = resolve(cs.candidate_id_strings(rows))
        parts = cs.partition_rows(rows, case_ids, set(meta) | case_ids)
        case_rows, hist_rows = parts["case"], parts["historical"] + parts["mixed"] + parts["unattributed"]
        case_truncated = hist_truncated = truncated
        if truncated:
            notes.append(f"The unscoped query returned more than {_POST_FILTER_FETCH_CAP} rows; case-local "
                         f"rows beyond that point were not seen. Treat this answer as INCOMPLETE.")
        if parts["mixed"] or parts["unattributed"]:
            notes.append(f"{len(parts['mixed'])} row(s) mix member and non-member alerts and "
                         f"{len(parts['unattributed'])} row(s) name no alert; both are shown under historical "
                         f"evidence because they are not restricted to this case.")

    case_total, hist_total = len(case_rows), len(hist_rows)
    if enforcement == "db_predicate" and case_truncated:
        notes.append(f"More than {cs.CASE_FETCH_CAP} case-local rows matched; only the first {cs.CASE_FETCH_CAP} were read.")
    if enforcement == "db_predicate" and hist_truncated:
        notes.append(f"More than {cs.HISTORICAL_FETCH_CAP} historical rows matched; only the first "
                     f"{cs.HISTORICAL_FETCH_CAP} (in the query's own order) were read, so the historical rows "
                     f"shown may not be the most recent ones.")

    # Limit AFTER scoping. Case rows diversify the bounded fetched pool,
    # then get the same recency re-rank as before;
    # historical rows are ranked by recency first so the background shown
    # is the most recent activity before the reference time.
    from src.retrieval.representative import representative_rows
    case_shown = representative_rows(case_rows, cs.CASE_TOP_K)
    if len(case_rows) > cs.CASE_TOP_K:
        notes.append('Case excerpts are diversity-selected from fetched rows; this is not an exhaustive timeline. '
                     'Selection covers activity, host, source and time where those fields were returned.')
    if reference_time:
        case_shown = weight_and_sort_records(case_shown, reference_time)
        hist_rows = weight_and_sort_records(hist_rows, reference_time)
    hist_shown = hist_rows[:cs.HISTORICAL_TOP_K]

    meta = resolve(cs.candidate_id_strings(case_shown + hist_shown))
    known = set(meta) | case_ids
    # Defence in depth: a row in the case channel that names a non-member
    # alert means the rewrite missed a pattern. Move it, and say so.
    leaked = [r for r in case_shown if any(a not in case_ids for a in cs.row_alert_ids(r, known))]
    if leaked:
        case_shown = [r for r in case_shown if r not in leaked]
        hist_shown = leaked + hist_shown
        notes.append(f"Post-check moved {len(leaked)} row(s) naming non-member alerts out of the case channel.")
    return cs.build_scoped_result(
        source="cypher", scope=case_scope, case_rows=case_shown, case_total=case_total,
        case_truncated=case_truncated, historical_rows=hist_shown, historical_total=hist_total,
        historical_truncated=hist_truncated, alert_meta=meta, enforcement=enforcement,
        queries=queries, notes=notes)


def query_cypher(question: str, query_timestamp: Optional[str] = None, case_scope: Optional[dict] = None) -> dict:
    """
    Generate and run a Cypher query against the graph database.
    Use this for complex questions requiring structured data, aggregations, or specific graph traversals
    Returns the query and the result context.

    `query_timestamp`, when given, is also appended to `question` as a
    "Reference time" note (rule 13 in cypher_generation_template above) so
    the LLM anchors relative-time phrases like "the last 24 hours" to the
    investigation's own reference point instead of real wall-clock
    `datetime()` -- this data is historical, so wall-clock "now" would never
    match it. Separately, it re-ranks the result rows by recency
    (src/retrieval/temporal.py) before they're returned -- the generated
    Cypher's RETURN clause is LLM-controlled and not fixed-schema, so rows
    are heuristically scanned for an ISO-8601 timestamp value rather than
    assuming a specific column.

    `case_scope` (state['case_scope'], set for case investigations) switches
    to case-scoped retrieval: the case filter is applied inside Neo4j
    BEFORE any limit, and non-member alerts come back in a separate,
    labelled historical channel. Without it (src/run.py, per-alert eval)
    behaviour is unchanged.
    """
    print(f"--- Executing Cypher Search for: {question} ---")
    enriched_question = question
    if query_timestamp:
        enriched_question = f'{question}\n\n(Reference time for relative time phrases: "{query_timestamp}")'
    if case_scope:
        # Case investigation: scope first, limit second, two labelled
        # channels. `context` is a structured dict here (see
        # run_scoped_cypher); dispatch_retrieval_node renders it.
        base = f"{enriched_question}\n\n{_SCOPE_NOTE}\n{_entity_hint(case_scope)}"

        last = {"query": ""}   # the query an exception belongs to, for the repair prompt

        def attempt(prompt: str):
            generated = last["query"] = _generate_cypher(prompt)
            print(f"Generated Cypher (before case scoping):\n{generated}")
            if not generated:
                raise ValueError("Cypher generation produced no valid query for this question")
            placeholder = _find_placeholder(generated)
            if placeholder:
                raise ValueError(f"The generated query anchors on a placeholder identifier ({placeholder!r}), "
                                 f"not on a real entity; its result would be empty by construction")
            return generated, run_scoped_cypher(generated, case_scope)

        rejected = []
        prompt = base
        for n in range(_GENERATION_ATTEMPTS):
            try:
                generated, context = attempt(prompt)
                break
            except Exception as exc:
                if not _repairable(exc) or n == _GENERATION_ATTEMPTS - 1:
                    raise
                error = f"{type(exc).__name__}: {exc}"
                rejected.append(error)
                # Each regeneration sees only the latest rejected query and why.
                prompt = f"{base}\n\n{_repair_feedback(last['query'], error)}"
        if rejected:
            context.setdefault("notes", []).append(
                f"{len(rejected)} generated quer{'y was' if len(rejected) == 1 else 'ies were'} rejected "
                f"({rejected[-1][:200]}); this is the regenerated one.")
        else:
            filters = _guessed_filters(generated) if _is_empty(context) else []
            if filters:
                try:
                    retry_query, retry_context = attempt(f"{base}\n\n{_empty_feedback(generated, filters)}")
                except Exception:
                    retry_query = retry_context = None
                if retry_context is not None and not _is_empty(retry_context):
                    retry_context.setdefault("notes", []).append(
                        f"The first query returned nothing after filtering on {'; '.join(filters)[:200]}; "
                        f"it was regenerated without the guessed filter.")
                    generated, context = retry_query, retry_context
                elif isinstance(context.get("notes"), list):
                    context["notes"].append(
                        f"This empty result came from a query that filtered on {'; '.join(filters)[:200]}; "
                        f"a broader retry also found nothing or failed.")
        return {"query": generated, "context": context}
    response = _get_cypher_qa_chain().invoke({"query": enriched_question})
    context = response["intermediate_steps"][1]["context"]
    if query_timestamp and isinstance(context, list):
        context = weight_and_sort_records(context, query_timestamp)
    return {
        "query": response["intermediate_steps"][0]["query"],
        "context": context
    }
