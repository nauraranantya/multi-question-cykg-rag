# src/agents/question_generation_agent.py
"""
Generates a small set of distinct investigative questions from a single
alert, instead of assuming one fixed interpretation of it. The same alert
(e.g. "Authentication success from a new IP") can be part of very different
real scenarios -- routine remote work, credential stuffing, or one step in
an active lateral-movement chain -- and telling those apart requires asking
different, targeted questions and checking the evidence for each, not
treating the alert as having a single obvious meaning.

Question taxonomy grounded in established SOC investigative practice:
- Diamond Model of Intrusion Analysis (Caltagirone, Pendergast & Betz, 2013,
  DoD technical report) -- Adversary / Capability / Infrastructure / Victim
  as the four axes an intrusion event can be pivoted on.
- Standard alert-triage question categories -- scope/asset criticality,
  temporal baseline, correlation with other activity, historical
  recurrence, threat-intel corroboration, attack-stage severity (see e.g.
  CyberDefenders' alert-triage guide and Prophet Security's alert-type
  investigation checklists).

Each generated question is tagged with a `source` using fixed routing
rules stated directly in the prompt below -- not a computed EIG/cost
estimate. See ../../defengraph_vs_agcyrag_analysis.md and PROTOTYPE.md for
why a rule-based policy was chosen here over a learned/computed router:
structural, correlational, temporal, or historical-recurrence questions
about *this* environment's own data route to Cypher; questions about the
general meaning, mitigation, or threat-intel context of a technique/CVE
route to the CSKG; anything needing free-text narrative matching not
captured by the graph's structured fields routes to Vector.

This also removes the need for src/agents/reflection_agent.py's reactive
rephrase-and-retry: rather than firing one question, discovering it failed,
and rewording it, the system asks several well-targeted questions up front.
An empty/negative result on one of them is itself an answer (it rules out
that hypothesis), not a failure to recover from.
"""
from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import llm, fast_llm
from src.config.structured import structured_model


class GeneratedQuestion(BaseModel):
    hypothesis: str = Field(
        description="The specific scenario this question is meant to confirm or rule out, e.g. "
        "'benign routine login', 'credential-stuffing / brute force', 'part of an active "
        "lateral-movement chain'. Always include at least one benign hypothesis alongside any "
        "malicious ones -- the same alert can be either."
    )
    question: str = Field(
        description="One sharp, answerable investigative question whose result would confirm or "
        "rule out the hypothesis above."
    )
    source: Literal["cypher", "vector", "cskg"] = Field(
        description=(
            "Which retrieval source answers this question, per fixed routing rules: 'cypher' for "
            "structural/correlational/temporal/historical-recurrence questions about this "
            "environment's own alert graph (other alerts on this host, prior activity from this "
            "IP, whether this user/host pair has been seen before); 'cskg' for questions about the "
            "general meaning, tactic, or mitigation of a MITRE technique/CVE/CWE/CAPEC; 'vector' "
            "only for free-text narrative matching the graph's structured fields can't capture. "
            "Prefer 'cypher' over 'vector' whenever the question could be answered from structured "
            "data -- alert data here is structured, not narrative documents."
        )
    )
    rationale: str = Field(description="One sentence: why this question discriminates the hypothesis.")


class QuestionSet(BaseModel):
    questions: List[GeneratedQuestion] = Field(
        description="3 to 5 questions for a broad investigation, covering more than one hypothesis "
        "-- never collapse to a single assumed interpretation of the alert. Exactly 1 question when "
        "the human message below says this is a narrow, specific follow-up (direct_question mode) "
        "instead."
    )


question_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a SOC analyst forming a differential diagnosis for a security activity case,
before committing to one interpretation of it.

Use the activity-group counts and diverse member excerpts to cover different behaviors in the
case. Do not anchor every hypothesis on the first alert or the highest-volume group. Include a
question about corroborating or contradictory evidence for any proposed attack explanation.
Treat attack names as alternatives to investigate, not conclusions. Do not ask whether earlier
activity was resolved as benign or authorized: this store contains alerts, not analyst verdicts.
MITRE descriptions explain possibilities; they cannot confirm occurrence in this environment.

The same alert can be part of very different real situations depending on context -- e.g. an
"authentication success from a new location" alert could be a traveling employee, a
credential-stuffing hit, or one step inside an active intrusion. Your job is NOT to decide which
one it is yet. Your job is to propose the plausible hypotheses and, for each, the single sharpest
question whose answer would confirm or rule it out.

Ground every question in one of these investigative angles (Diamond Model of Intrusion Analysis +
standard alert-triage practice):
- Victim / scope: what asset is affected, and how critical is it?
- Infrastructure: what IPs/domains/hosts are involved, and are they linked to other activity?
- Capability: what technique/tool/behavior does this alert actually represent?
- Adversary: is there any attribution or campaign signal?
- Temporal baseline: is this normal timing/behavior for this host or user?
- Correlation: are other hosts, users, or alerts showing related activity?
- Historical recurrence: has this fired before, on this entity, and was it resolved as benign?
- Threat-intel corroboration: does this match a known technique, CVE, or IOC?
- Attack-stage severity: if malicious, is this early recon, or already lateral movement/exfiltration?

Rules for tagging each question's `source` (apply these directly, do not estimate cost or value):
1. If it needs correlation, timing, aggregation, or recurrence over THIS environment's own alert
   graph -> 'cypher'.
2. If it needs the general meaning, tactic, or mitigation of a technique/CVE/CWE/CAPEC -> 'cskg'.
3. If it needs free-text matching against raw log narrative that structured fields won't
   capture -> 'vector'.
4. When in doubt between 'cypher' and 'vector', prefer 'cypher' -- alert data here is structured.
5. If the evidence names a software product or component together with a version on the TARGET
   side (e.g. in a requested URL path, a server banner, a package or plugin name), and a known
   vulnerability in it would change the assessment, ask one 'cskg' question of the form "Which CVEs
   affect <product> <version>, and what are their CVSS v3 scores?", with the product name and version
   copied exactly from the evidence. Do not ask it for the client side (a scanner's or browser's user
   agent identifies the tool used, not a vulnerable component), and never ask cskg whether a CVE was
   exploited here -- only the environment's own alerts can show that.

What each source can and cannot actually answer -- a question outside a source's real coverage
will just come back empty, wasting that hypothesis slot on an unfalsifiable question instead of a
genuinely discriminating one:
- 'cypher' (this environment's own graph) ONLY has: Host/IP/User/Alert/MitreTechnique nodes,
  connected by which host an alert fired on, its source/destination IP, targeted user, and any
  natively-tagged MITRE technique, each alert carrying id/title/rule_level/timestamp/full_log.
  It can answer correlation, recurrence, and timing questions about THESE entities (has this
  host/IP/user triggered other alerts, what technique history exists, what happened before/after).
  It has NO data on: whether activity was scheduled/authorized, IT/security team activity logs,
  configuration or IDS-rule change history, asset ownership/criticality, or anything about
  whether something was "legitimate" -- don't ask cypher questions whose answer would require any
  of that; the graph has no node or property for it, so the answer is always going to be empty,
  not because of a bad query.
- 'cskg' (public MITRE ATT&CK/CVE/CWE/CAPEC reference data ONLY) can explain what a technique/CVE
  IS, its tactic, and its documented mitigations. It has NO IP addresses, hostnames, live threat
  feeds, IOC/reputation data, or anything about THIS environment specifically -- never ask a cskg
  question that needs correlating a specific IP/host/user against threat intelligence (e.g. "is
  this IP known malicious") or checking whether a technique was actually used here -- cskg can
  only speak to what a technique means in general, not whether this alert's traffic constitutes
  it. When a question references a technique, prefer asking by its full name if you have one
  (e.g. "Network Service Discovery") over just its ID (e.g. "T1046") -- most cskg tools match
  against the technique's name text, not its ID.
- 'vector' free-text search over the same alert data cypher has (not a separate corpus) -- only
  useful when the question needs matching raw log narrative text that structured fields (host,
  IP, user, rule_level) can't express.

Always include at least one benign/expected-activity hypothesis. Produce 3 to 5 questions total."""
    ),
    (
        "human",
        "Alert:\n{alert_summary}\n\n"
        "Case context (static fact sheet for the clustered investigation case this belongs to, "
        "if any -- otherwise 'N/A'):\n{case_context}\n\n"
        "Conversation so far (prior turns in this investigation, if this is a follow-up question -- "
        "otherwise 'N/A'). If there's a real conversation history, prefer hypotheses/questions that "
        "build on or narrow down what's already been asked and found, rather than re-deriving a "
        "generic hypothesis set from scratch:\n{conversation_context}\n\n"
        "{mode_instruction}\n\n"
        "Generate the discriminating question(s)."
    ),
])

question_generation_chain = question_generation_prompt | structured_model(llm, QuestionSet)
# Cost-tier routing (see src/config/settings.py::fast_llm) -- direct_question
# turns are a single narrow follow-up, not a differential-diagnosis task, so
# generate_questions() below routes them to the cheaper model.
question_generation_chain_fast = question_generation_prompt | structured_model(fast_llm, QuestionSet)

# One interpretation uses the SAME full model as the multi-hypothesis arm.
# Reuse the source capability rules but replace differential-diagnosis directives.
_single_system = question_generation_prompt.messages[0].prompt.template
_single_rules = _single_system[_single_system.index("Rules for tagging"):]
_single_rules = _single_rules[:_single_rules.index("Always include at least one benign")]
single_interpretation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a SOC analyst. Commit to the single most likely explanation of this "
     "activity case. Generate exactly ONE targeted question to investigate that explanation. "
     "Do not generate alternative hypotheses or extra mitigation questions.\n\n" + _single_rules),
    ("human", "Case: {alert_summary}\nEvidence: {case_context}\nPrior conversation: {conversation_context}"),
])

class SingleQuestionSet(BaseModel):
    questions: List[GeneratedQuestion] = Field(min_length=1, max_length=1)

single_interpretation_chain = single_interpretation_prompt | structured_model(llm, SingleQuestionSet)

_MODE_INSTRUCTIONS = {
    "full_hypothesis": (
        "Mode: full_hypothesis. Produce the full 3-to-5-question hypothesis set as usual, covering "
        "more than one plausible interpretation."
    ),
    "direct_question": (
        "Mode: direct_question. This is a narrow, specific follow-up in an ongoing investigation, "
        "not a broad differential-diagnosis situation. Produce exactly ONE question -- the single "
        "sharpest, most direct query that retrieves what's needed to answer it. Do not invent "
        "multiple speculative hypotheses for a question this specific; a single (hypothesis, "
        "question, source) entry covering the follow-up itself is enough."
    ),
}


def generate_questions(
    alert_summary: str,
    case_context: str = "N/A",
    conversation_context: str = "N/A",
    mode: str = "full_hypothesis",
) -> QuestionSet:
    """`alert_summary` is the same narrative rendering used at ingest time
    (see src/ingestion/graph_loader.py::_alert_narrative) for auto-triggered
    alerts, or the analyst's raw question text for manually-typed queries.
    `case_context`/`conversation_context` are only set for the human-invoked
    investigation flow (src/investigation/) -- see src/graph/workflow.py's
    question_generation_node. `mode` is guardrails_agent.py's
    investigation_mode decision ('answer_from_context' never reaches here --
    question_generation_node skips this call entirely in that case)."""
    if mode == "single_interpretation":
        return single_interpretation_chain.invoke({
            "alert_summary": alert_summary, "case_context": case_context,
            "conversation_context": conversation_context})
    chain = question_generation_chain if mode == "full_hypothesis" else question_generation_chain_fast
    return chain.invoke({
        "alert_summary": alert_summary,
        "case_context": case_context,
        "conversation_context": conversation_context,
        "mode_instruction": _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS["full_hypothesis"]),
    })
