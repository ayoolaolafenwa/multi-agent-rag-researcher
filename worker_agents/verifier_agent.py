from .model_runner import run_model

VERIFIER_MODEL = "gpt-5.4"
VERIFIER_REASONING_EFFORT = "low"

"""
Verifier Agent
================================================
Verifier agent used by the orchestrator to verify the report written
by the writer agent.

It checks the content and claims in the draft against the evidence
before returning the final verified report.
"""
def verifier_agent(
    user_query: str,
    written_draft: str,
    evidence_text: str,
    verbose: bool = False,
) -> str:
    if verbose:
        print("[Verifier Agent] Verifying report...")

    instructions = (
        """
        Verify the draft against the evidence and the user query, then return only the final answer.
        Start with the answer and make sure it directly answers the user's question.
        Remove anything that is supported by the evidence but does not answer the query.
        No preamble, review notes, or meta lead-ins such as "Below is", "Here is", or "Based on the cited sources".
        Keep the writing concise, natural, and complete enough to fully answer the question.
        Preserve useful supporting detail when it helps answer the question more completely.
        When the evidence supports it, keep the most important comparisons, caveats, and specific facts instead of collapsing the answer into a minimal summary.
        For judgments, comparisons, recommendations, or conclusions, state the best-supported conclusion clearly if supported.
        For short follow-ups, answer briefly and directly.
        Keep only supported statements.
        Add citations at the end of supported sentences.
        Use the exact citation field from Document evidence for PDF citations.
        Use the exact title and exact URL from Web evidence for web citations.
        When citing web evidence, use Markdown links in the form [Exact Source Title](Exact URL).
        If the final answer uses web evidence, do not omit the URL citations.
        If the evidence is weak, incomplete, or not enough for a confident conclusion, say so.
        End the answer immediately after the last cited sentence.
        """
    )

    # Pass the query, report draft, and evidence together so the verifier can
    # check the claims and return the final verified report.
    input_text = (
        f"User query: {user_query}\n\n"
        f"Report draft:\n{written_draft}\n\n"
        f"Evidence:\n{evidence_text}"
    )

    response = run_model(
        instructions=instructions,
        input_data=input_text,
        model=VERIFIER_MODEL,
        reasoning_effort=VERIFIER_REASONING_EFFORT,
        tools=None,
    )
    return response.output_text
