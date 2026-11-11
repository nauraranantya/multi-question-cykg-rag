# src/chains/synthesizer.py
"""
`cited_entities`/`mitre_techniques` are read by src/agents/grounding_check.py
to verify the report only references entities actually present in the
retrieved context.

Also now consumes `hypothesis_summary` (built from question_generation_agent's
QuestionSet + review_agent's per-question verdicts, rendered in
src/graph/workflow.py::_render_hypothesis_summary) -- a deterministic
rollup of which candidate scenarios were confirmed/ruled out/left
inconclusive. This absorbs log_analysis_agent.py's old summarization role
(that module is dissolved -- its other job, deciding whether a CSKG lookup
was needed, is now decided per-question by question_generation_agent's
`source` tag instead of after the fact).
"""
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import llm, fast_llm
from src.config.structured import structured_model


class ReportClaim(BaseModel):
    """One checkable statement. src/agents/grounding_check.py validates
    every claim against the alerts it cites (existence, case membership,
    entity presence). Whether those alerts actually support the claim is
    rated here, by the synthesizer itself, in `support` (see SELF-VALIDATION
    in the prompt below). That rating is used as is by default;
    EVIDENCE_CLAIM_VERIFIER=llm has a separate verifier audit it instead."""
    claim: str = Field(description="One atomic factual statement that final_answer or critical_analysis asserts, "
                       "in a single sentence. State what the cited alerts show, no more.")
    scope: Literal["case", "historical", "knowledge_base", "not_found"] = Field(
        description="'case' if the statement is about this case's own activity and rests on CASE-LOCAL alerts "
        "(or member alerts listed in the Case Context); 'historical' if it is about other or earlier activity "
        "and rests on HISTORICAL EVIDENCE; 'knowledge_base' if it is general MITRE/CVE knowledge taken from "
        "the Cybersecurity Knowledge Base Context; 'not_found' if it is a negative finding -- that the "
        "retrieved evidence contains no sign of something."
    )
    evidence_alert_ids: List[str] = Field(
        default_factory=list,
        description="Alert IDs, copied exactly from the context, whose own content supports this claim. "
        "Required for 'case' and 'historical' claims; leave empty for 'knowledge_base' and 'not_found' claims.",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="The hosts, IPs, users, domains or technique IDs this claim is about, exactly as they "
        "appear in the cited alerts.",
    )
    # Reason before rating, so the rating is written after the alerts have
    # been re-read rather than before.
    support_reason: str = Field(
        default="",
        description="One sentence naming what the cited alerts do and do not show about this claim, judged "
        "from those alerts alone. 'Not applicable.' for 'knowledge_base' and 'not_found' claims.",
    )
    support: Literal["supported", "partially_supported", "unsupported", "contradicted", "not_applicable"] = Field(
        default="not_applicable",
        description="Your own check of this claim against ONLY the alerts in evidence_alert_ids (see "
        "SELF-VALIDATION). Required for 'case' and 'historical' claims: 'supported' if those alerts directly "
        "show what the claim says; 'partially_supported' if the claim goes further than they do; 'unsupported' "
        "if they do not show it; 'contradicted' if they show the opposite. 'not_applicable' only for "
        "'knowledge_base' and 'not_found' claims.",
    )


class SynthesizedReport(BaseModel):
    maliciousness: Literal['benign', 'suspicious', 'malicious', 'undetermined'] = Field(
        default='undetermined', description='Assessment of this case, separate from the decision to escalate. '
        'Malicious requires case-local behavior establishing an attack, not merely a detector or MITRE tag.')
    uncertainty: List[str] = Field(default_factory=list,
        description='Missing evidence, competing explanations, retrieval gaps, and limits of the sampled context.')
    escalation_rationale: str = Field(default='',
        description='Why to escalate, monitor, or ignore based on supported observations and potential impact. '
        'Escalation can request human investigation without asserting a confirmed attack.')
    original_question: str = Field(description="The user's original question, restated.")
    cypher_context_summary: str = Field(description="Summary of the Cypher/graph query context, or 'Not applicable for this query.'")
    vector_context_summary: str = Field(description="Summary of the vector/full-text search context, or 'Not applicable for this query.'")
    generated_question_for_rdf: str = Field(description="The question generated for the cybersecurity knowledge base, or 'Not applicable for this query.'")
    cskg_context_summary: str = Field(description="Summary of the cybersecurity knowledge base (RDF/MITRE) context, or 'Not applicable for this query.'")
    critical_analysis: str = Field(description="How the information from all sources connects; whether a single source is sufficient on its own.")
    contextual_linkage: str = Field(description="The logical flow of the investigation, e.g. how a log finding led to a knowledge-base lookup.")
    final_answer: str = Field(description="The final, well-structured, human-readable answer for the analyst.")
    cited_entities: List[str] = Field(
        default_factory=list,
        description="Every specific entity identifier referenced in critical_analysis/final_answer "
        "(hostnames, IPs, usernames, ticket/document names) -- used to verify the answer is grounded "
        "in the retrieved context, not invented.",
    )
    mitre_techniques: List[str] = Field(
        default_factory=list,
        description="MITRE ATT&CK technique IDs (e.g. 'T1110') referenced in the analysis.",
    )
    claims: List[ReportClaim] = Field(
        default_factory=list,
        description="Every factual statement final_answer and critical_analysis rely on, one entry each, with "
        "the alert IDs that support it. Any alert ID mentioned anywhere in the report must appear in some "
        "claim's evidence_alert_ids. Leave empty only for a pure general-knowledge answer with no alert data.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in final_answer given how much of the available evidence actually supports it. "
        "Note: this is capped downward by src/graph/workflow.py based on how many of the 3 context sources "
        "(Cypher/vector/CSKG) actually returned substantive data -- don't default to 'high' out of habit; a "
        "genuinely low-evidence answer should be rated 'low' or 'medium' here regardless of the cap."
    )
    recommended_priority: Literal["ignore", "monitor", "escalate"] = Field(
        description="Triage recommendation for a SOC analyst: 'escalate' if this warrants immediate "
        "human investigation/response, 'monitor' if it's worth tracking but not urgent, 'ignore' if it's "
        "benign/expected activity or there's no log/alert question to triage at all (e.g. a general "
        "cybersecurity-knowledge question)."
    )
    mitigation_suggestions: List[str] = Field(
        default_factory=list,
        description="Concrete, actionable response/mitigation steps a SOC analyst could take given "
        "final_answer -- e.g. specific accounts to disable, hosts to isolate, rules to tighten. Empty "
        "if recommended_priority is 'ignore' or there's nothing actionable to suggest (e.g. a general "
        "knowledge question)."
    )


synthesis_prompt = ChatPromptTemplate.from_template("""You are an expert cybersecurity analyst creating a final report.
Your task is to synthesize information gathered across several investigative questions -- each
generated to confirm or rule out a specific hypothesis about this alert -- into one report.

Separate (1) directly observed behavior, (2) inferred explanation and alternatives, and (3) action.
Use maliciousness and uncertainty explicitly. Escalation is a request for human investigation,
not a declaration that an attack is proven; state its rationale independently. Do not describe a
whole case as primarily benign or malicious from a few excerpts. Use the activity-group counts
to describe its composition, and acknowledge unexamined groups. A large cluster can contain
unrelated benign and suspicious behaviors at once.
MITRE technique descriptions are general reference knowledge, NEVER confirmation that a technique
occurred here. Keep detector-tagged, hypothesized and behaviorally supported techniques distinct.
Do not infer a specific tool, intent, successful compromise, or causal chain from a nonspecific
anomaly. A resource-usage spike has several possible causes; a protocol error does not establish
interception. Negative findings apply only to the retrieved evidence, not the entire environment.
Write final_answer in the ANSWER FORMAT given at the end of this prompt.

If a section's source has no data (e.g. no log data was queried), state "Not applicable for this query." for that field.

CRITICAL: only put entities (hostnames, IPs, usernames, MITRE technique IDs, etc.) into cited_entities/mitre_techniques
or reference them in critical_analysis/final_answer if they literally appear in the context below. Do not invent
or infer entities that are not present in the provided context.

EVIDENCE SCOPE: when this is a case investigation, retrieved context is split into two channels.
"CASE-LOCAL EVIDENCE" holds alerts that are members of this case; the member alerts listed in the Case
Context are case-local too. "HISTORICAL EVIDENCE" holds other alerts from the environment and is background
only. Anything you conclude about this case -- which hosts or IPs are involved, which technique was used,
whether the activity is malicious, how urgent it is -- must rest on case-local alerts. Historical alerts
may support only statements that are explicitly about other or earlier activity (recurrence, baseline),
and you must describe them as outside this case. Never name a host, IP or alert as involved in this case
on the strength of historical evidence alone.

CLAIMS: fill `claims` with every factual statement your final_answer and critical_analysis rely on, one
per entry, each with the alert IDs whose own content supports it and the scope those alerts belong to.
Every alert ID you mention anywhere must appear in some claim's evidence_alert_ids. Each claim is checked
afterwards against exactly the alerts it cites: write claims that state what those alerts show. If you
cannot tie a statement to a specific alert, do not present it as established -- say it is unverified.
A negative finding ("no alert in this case indicates lateral movement") is a 'not_found' claim with no
alert IDs; it records that the retrieved evidence was silent, not that the activity did not happen.

SELF-VALIDATION: check whether your claims follow from their evidence before you finish. A separate
verifier may audit these claims; your self-rating is not independent confirmation. For each 'case' or
'historical' claim, re-read ONLY the alerts it cites -- not the case fact sheet,
not the hypothesis summary, not the rest of your draft -- and decide whether those alerts, by themselves,
establish the claim. Write what they do and do not show in support_reason, then set support:
- supported: the cited alerts directly state or show what the claim says.
- partially_supported: they show something related, but the claim goes further than they do -- for example
  it asserts intent, a technique, a cause, a scope or a degree of certainty that the alerts do not show.
- unsupported: they do not show it.
- contradicted: they show the opposite.
Use nothing except the cited alerts: not background knowledge, not what is typical, not what the claim's
wording suggests. A record of a detector or rule firing shows that it fired on that value; whether the
underlying activity is malicious is a separate claim that needs alerts showing it. Whether an alert belongs
to this case is settled by the section it appears in (CASE-LOCAL, Case Context or HISTORICAL), so do not
expect an alert's own text to mention the case. Judge strictly, and when undecided between two levels
choose the lower.
If a claim does not come out 'supported', first restate it as what the cited alerts actually show and rate
that instead. If you still keep a claim that is not 'supported', rate it as it is: final_answer and
critical_analysis must then present it as unverified, and confidence and recommended_priority must rest
only on the claims you rated 'supported'. These ratings decide how much the report is allowed to assert, so
rate accurately rather than favourably. Every 'case' or 'historical' claim needs a rating; set support to
'not_applicable' only for 'knowledge_base' and 'not_found' claims.

RETRIEVAL FAILURES: if a source is listed under "Retrieval failures" below, it could not be queried at
all. Say so in that source's summary field (do not write "Not applicable"), do not treat the missing
result as a negative finding, and do not fill the gap from your own knowledge.

Use the hypothesis summary below as your primary reasoning scaffold: weigh which hypotheses were
confirmed, ruled out, or left inconclusive, and let that drive critical_analysis and final_answer --
don't just restate raw context. A hypothesis being ruled out is as informative as one being confirmed.

Also set recommended_priority: 'escalate' for genuinely suspicious/malicious activity needing prompt human
response, 'monitor' for activity worth tracking but not urgent, 'ignore' for benign/expected activity or
questions that aren't about a specific alert at all (e.g. general cybersecurity knowledge questions).
Populate mitigation_suggestions with concrete response actions when recommended_priority is 'monitor' or
'escalate'; leave it empty otherwise.

ANSWER FORMAT (final_answer only; every other field is filled as described above):
{answer_format}

**Case Context (static case fact sheet, if this belongs to a clustered investigation case):**
{case_context}

**Conversation So Far (prior turns in this investigation, if any):**
{conversation_context}

**Original Question:**
{original_question}

**Hypothesis Summary (generated questions + review verdicts):**
{hypothesis_summary}

**Cypher Log Information Context:**
{log_cypher_context}

**Vector Log Information Context:**
{log_vector_context}

**Generated Question for Cybersecurity Knowledge Base:**
{generated_question_for_rdf}

**Cybersecurity Knowledge Base Context (from RDF Agent):**
{mcp_rdf_context}

**Retrieval failures (sources that could not be queried this investigation):**
{retrieval_failures}
{retry_note}
""")

# final_answer layout. The initial report is read once, to triage: verdict
# first, then evidence, then what is open, then actions. A follow-up is a
# question inside a conversation: it gets a direct answer, not a new report.
# The structured fields (claims, maliciousness, priority, ...) are produced
# in both cases, since validation and scoring read them, not the prose.
INITIAL_ANSWER_FORMAT = """Markdown, at most about 350 words, in exactly these parts:
1. One or two sentences, no heading: the bottom line -- what this case most likely is, how sure you are,
   and whether a human should look now. Do not restate the priority/confidence field values as labels.
2. "**What the alerts show**": 3 to 6 bullets of directly observed behavior, in time order, each ending
   with its alert ID(s) in parentheses. Observations only, no interpretation.
3. "**Assessment**": the most likely explanation and the alternatives still open, and which hypotheses
   were ruled out and on what evidence. Mark anything not 'supported' as unverified.
4. "**Gaps**": bullets for what could not be checked (failed sources, invalid queries, unexamined
   activity groups, truncated results). Write "None." if there are none.
5. "**Actions**": at most 5 numbered actions, case-specific ones first (named hosts, accounts, IPs),
   general hardening last. Omit this part if recommended_priority is 'ignore'."""

FOLLOWUP_ANSWER_FORMAT = """This is a follow-up question inside an ongoing investigation. Answer it directly, in plain
prose: no headings, no bullet lists unless the question asks for a list, at most about 200 words.
The first sentence answers the question (yes/no/what/which). Then give the evidence for it, citing
alert IDs in parentheses, and say whether it changes the earlier assessment. If the question could not
be answered (no evidence, failed or invalid query), say so plainly and why. Do not restate the case,
and mention actions only if this answer changes them."""


def answer_format(is_followup: bool) -> str:
    return FOLLOWUP_ANSWER_FORMAT if is_followup else INITIAL_ANSWER_FORMAT


synthesis_chain = synthesis_prompt | structured_model(llm, SynthesizedReport)
# Cost-tier routing (see src/config/settings.py::fast_llm) -- workflow.py's
# synthesize_node picks between these two based on investigation_mode.
synthesis_chain_fast = synthesis_prompt | structured_model(fast_llm, SynthesizedReport)
