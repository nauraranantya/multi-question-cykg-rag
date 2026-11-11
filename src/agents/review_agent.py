# src/agents/review_agent.py
"""
Reviews evidence retrieved for one investigative question against the
hypothesis it was generated to discriminate (see question_generation_agent.py)
-- not just whether *some* fact came back.

Previously this was a flat sufficient/insufficient check whose only purpose
was to gate an immediate rephrase-and-retry (reflection_agent.py). That
retry loop is dissolved: src/agents/question_generation_agent.py already
asks several distinct, well-targeted questions up front, so a single
question coming back empty is no longer treated as a failure to recover
from -- it's potentially itself the answer (e.g. "no prior alerts from
this IP" genuinely rules out a "known repeat offender" hypothesis). What
the system needs from review now is not "should we retry" but "what does
this evidence say about the hypothesis," so the synthesizer can weigh
confirmed/ruled-out/inconclusive verdicts across all questions at once.
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import llm
from src.config.structured import structured_model


class ReviewOutput(BaseModel):
    sufficient: bool = Field(
        description="Does the context contain at least one concrete, factual data point relevant "
        "to the question -- including a clear negative (e.g. 'zero results found')? Only false if "
        "the context is empty/erroneous in a way that gives no information either way."
    )
    hypothesis_verdict: Literal["confirmed", "ruled_out", "inconclusive"] = Field(
        description="Does this evidence confirm the hypothesis, rule it out, or remain inconclusive "
        "either way? Judge based on what the result -- including an empty one -- actually implies "
        "for THIS specific hypothesis, not just whether data was returned. E.g. zero prior alerts "
        "for a host is 'ruled_out' evidence against a 'recurring known issue' hypothesis, not "
        "'inconclusive' just because the list is empty."
    )
    reasoning: str = Field(description="One or two sentences justifying both fields.")


review_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a SOC analyst evaluating evidence gathered for one investigative question, given "
        "the specific hypothesis that question was meant to discriminate. Decide (a) whether the "
        "retrieved context contains a usable factual data point -- a clear negative counts -- and "
        "(b) whether it confirms, rules out, or leaves inconclusive the stated hypothesis. Do not "
        "default to 'inconclusive' just because the result is short or empty: reason about what "
        "that specific result implies for this specific hypothesis. "
        "The context may be divided into 'CASE-LOCAL EVIDENCE' (alerts that are members of the case "
        "under investigation) and 'HISTORICAL EVIDENCE' (other alerts in the environment). A "
        "hypothesis about what happened in this case can only be confirmed by case-local rows; "
        "historical rows can confirm or rule out only recurrence/baseline hypotheses, and an empty "
        "case-local section means the case itself contains no such evidence, whatever the historical "
        "section shows. A truncated or selectively retrieved section is not an exhaustive search: "
        "absence cannot rule out unobserved activity. A failed query is not negative evidence. "
        "Before ruling a hypothesis out, check that the retrieval could have returned the evidence: "
        "a result that only shows rows matched through a keyword filter, through a source/destination "
        "IP or user edge (many alerts carry the IP or user only in their log text), or a verdict "
        "column computed inside the query (e.g. a false/null 'happened' flag) is 'inconclusive' "
        "when the hypothesis is not ruled out by rows actually shown. "
        "A detector firing proves an observation, not its intent or cause. Resource-usage anomalies "
        "alone cannot distinguish workload, cracking, mining, or faults; protocol anomalies alone "
        "cannot establish interception or exfiltration. Public MITRE descriptions and mitigations "
        "never confirm a case hypothesis without case-local behavioral evidence."
    ),
    (
        "human",
        "Hypothesis: {hypothesis}\n\nQuestion: {question}\n\nContext:\n{context}\n\n"
        "Evaluate sufficiency and the hypothesis verdict.",
    ),
])
review_chain = review_prompt | structured_model(llm, ReviewOutput)
