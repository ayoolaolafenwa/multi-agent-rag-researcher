import json
import os
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel
from memory import infer_route_used
from qdrant_vector_database import get_indexed_document_catalog, similarity_search
from .model_runner import run_model
from tavily import TavilyClient

"""
Retriever Agent 
=====================================================================================
It used by the Orchestrator for information retrieval. 
It uses two tools: document retrieval and web search.

Given a query, it decides whether to retrieve evidence from the indexed PDFs,
search the web for up-to-date information or use both tools.

If local document evidence is weak or missing, it can fall back to web search
to gather broader context.
"""

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) 

# Structured output returned by the retriever agent.
class ResearchEvidencePack(BaseModel):
    query: str
    route_used: Literal["documents", "web", "both", "none"]
    summary: str
    document_evidence: Optional[Dict[str, Any]] = None
    web_evidence: Optional[Dict[str, Any]] = None


# Retrieve relevant document evidence for a query.
def retrieve_document(
    query: str,
    per_doc_topk: int = 4,
    score_threshold: Optional[float] = 0.2,
) -> Dict[str, Any]:
    try:
        results = similarity_search(
            query=query,
            per_doc_topk=per_doc_topk,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        return {
            "query": query,
            "summary": f"Document retrieval failed: {type(exc).__name__}",
            "chunks": [],
        }
    return {
        "query": query,
        "summary": (
            "Retrieved relevant evidence from the uploaded PDFs."
            if results else
            "No sufficiently relevant evidence was found in the uploaded PDFs."
        ),
        "chunks": [
            {
                "document_name": item["document_name"],
                "document_title": item.get("document_title") or item["document_name"],
                "page_number": int(item["page_number"]),
                "chunk_id": item["chunk_id"],
                "citation": item["citation"],
                "content": item["content"],
                "score": float(item["score"]),
            }
            for item in results
        ],
    }

# Search the web for supporting context.
def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    if tavily is None:
        return {"query": query, "results": []}

    try:
        result = tavily.search(
            query=query,
            search_depth="basic",
            max_results=num_results,
            include_answer=False,
            include_raw_content=True,
            include_images=False,
        )
        return {"query": query, "results": result.get("results", [])}
    except Exception:
        return {"query": query, "results": []}


RETRIEVER_MODEL = "gpt-5.4-mini"
RETRIEVER_REASONING_EFFORT = "low"

# Guides the retriever agent on how to interact with the available tools
# (document retrieval and web search).
RETRIEVER_TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "retrieve_document",
        "description": (
            """Search the indexed PDF corpus and return the most relevant chunks with
            document names, titles, page numbers, exact citation strings, and
            scores. Preserve the returned document names, titles, page numbers, and
            citation strings because they are needed downstream for accurate PDF
            citations in the final answer. Prefer this when the query is plausibly
            covered by the uploaded PDFs, is closely related to the indexed document
            titles or topics, when the user explicitly asks about the uploaded PDFs,
            or when document-grounded evidence is needed."""
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Rewrite the user's request into a self-contained search query "
                        "for the indexed PDFs. Include omitted subject details from "
                        "follow-up context when needed."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_search",
        "description": (
            """Search the web for recent, changing, external, or clearly non-PDF
            information and return concise results with exact source titles, exact
            URLs, and source metadata. Preserve the returned titles and URLs because
            they are needed downstream for accurate web citations in the final
            answer. Prefer this when the query does not match the indexed document
            titles or topics, when external or current evidence is needed, or after
            document retrieval is empty or insufficient."""
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Rewrite the user's request into a self-contained web search "
                        "query. Include omitted subject details from follow-up context "
                        "when needed."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

# Instructions guiding the LLM's behavior in retriever agent
RETRIEVER_INSTRUCTIONS = """
You are the retriever worker for a research assistant.

Indexed document titles and topic hints are provided in the input.

Rules:
- This worker is only called for evidence requests, so use at least one retrieval tool unless the query is empty or malformed.
- Use the indexed document titles/topics to decide whether the active PDFs are likely relevant.
- If the query does not appear related to the indexed document topics and the user is not explicitly asking about the PDFs, go straight to web_search.
- If retrieve_document returns no relevant chunks and the user is not explicitly asking about the PDFs, call web_search before finishing.
- If retrieve_document returns relevant chunks, do not call web_search unless newer or external evidence is still needed.
- Use the last user query only when the current query is a follow-up that depends on it.
- Avoid redundant tool calls unless a prior result was empty or clearly insufficient.
- After tool use, return a short retrieval summary, not a user-facing answer.
"""

def retriever_agent(
    user_query: str,
    *,
    last_user_query: str = "",
    verbose: bool = False,
) -> ResearchEvidencePack:
    
    previous_response_id: Optional[str] = None

    # Expose indexed document titles so the model can judge whether the PDFs
    # are relevant before choosing a retrieval tool.
    document_catalog = get_indexed_document_catalog()
    indexed_documents_text = "\n".join(
        f"- {item['title']} (file: {item['file_name']})"
        for item in document_catalog
    ) or "- None"
    pending_input: Any = (
        f"Current user query: {user_query.strip()}\n"
        f"Last user query: {last_user_query.strip() or 'None'}\n"
        f"Indexed document titles and topic hints:\n{indexed_documents_text}"
    )
    document_evidence: Optional[Dict[str, Any]] = None
    web_evidence: Optional[Dict[str, Any]] = None
    summary = ""

    # Allow a few tool-call rounds before finalizing the retrieval summary.
    for _ in range(4):
        response = run_model(
            instructions=RETRIEVER_INSTRUCTIONS,
            input_data=pending_input,
            model=RETRIEVER_MODEL,
            tools=RETRIEVER_TOOL_SCHEMAS,
            previous_response_id=previous_response_id,
            reasoning_effort=RETRIEVER_REASONING_EFFORT,
        )
        previous_response_id = response.id

        tool_results = []
        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            # If the model stops calling tools, treat its final text as the
            # retriever-facing summary for this turn.
            summary = (response.output_text or "").strip()
            break

        for call in function_calls:
            query = json.loads(call.arguments)["query"].strip()

            if call.name == "retrieve_document":
                if verbose:
                    print("[Retriever Agent] Retrieving document evidence...")
                function_response = retrieve_document(query)
                if function_response.get("chunks"):
                    document_evidence = function_response
            else:
                if verbose:
                    print("[Retriever Agent] Searching the web...")
                function_response = web_search(query)
                if function_response.get("results"):
                    web_evidence = function_response

            tool_results.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(function_response),
                }
            )

        pending_input = tool_results

    document_chunks = document_evidence.get("chunks") if document_evidence else []
    web_results = web_evidence.get("results") if web_evidence else []
    route_used = infer_route_used(document_chunks or [], web_results or [])

    # Return the route used, a short summary and any collected evidence.
    return ResearchEvidencePack(
        query=user_query,
        route_used=route_used,
        summary=summary,
        document_evidence=document_evidence if document_evidence and document_evidence.get("chunks") else None,
        web_evidence=web_evidence if web_evidence and web_evidence.get("results") else None,
    )