from .model_runner import run_model

WRITER_MODEL = "gpt-5.4"
WRITER_REASONING_EFFORT = "low"

"""
Writer Agent 
=====================================================================================
It used by the Orchestrator to write a report on the retrieved information
from the retriever agent.
Main role is to draft a clear, grounded response from the evidence it receives.
"""
def writer_agent(user_query: str, evidence_text: str, verbose: bool = False) -> str:
    if verbose:
        print("[Writer Agent] Writing report...")

    instructions = (
        """
        Answer the user using only the evidence.
        Start with the answer.
        Be clear, complete, and concise.
        Include the key supporting details needed to fully answer the question.
        When the evidence supports it, include the most important comparisons, caveats, or specific facts rather than giving a minimal summary.
        For judgments, comparisons, recommendations, or conclusions state the best-supported conclusion clearly.
        For short follow-ups, answer briefly and directly.
        Do not add unsupported facts.
        Use the exact citation field from Document evidence for PDF citations.
        If web evidence is present, synthesize it into a concise, self-contained answer and include explicit web citations with the exact title and exact URL from Web evidence.
        When citing web sources, use Markdown links in the form [Exact Source Title](Exact URL).
        Do not omit web source URLs when web evidence is used.
        If the evidence is weak or incomplete, say so.
        Do not end with a question, a suggestion for the user to ask a follow-up, or an offer for more help.
        """
    )

    # Pass the user query together with the retrieval context for drafting.
    input_text = (
        f"User query: {user_query}\n\n"
        f"Evidence:\n{evidence_text}"
    )

    response = run_model(
        instructions=instructions,
        input_data=input_text,
        model=WRITER_MODEL,
        reasoning_effort=WRITER_REASONING_EFFORT,
        tools=None,
    )
    return response.output_text
