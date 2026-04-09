import json
from typing import Any

# infer route used from available document and web evidence
def infer_route_used(document_chunks: list[Any], web_results: list[Any]) -> str:
    if document_chunks and web_results:
        return "both"
    if document_chunks:
        return "documents"
    if web_results:
        return "web"
    return "none"


# build evidence context from saved evidence json
def build_evidence_context(
    evidence_json: str,
    *,
    include_formatted_evidence: bool = False,
) -> dict[str, Any]:
    if not evidence_json:
        return {"has_evidence": False, "summary": "None", "formatted_evidence": ""}

    try:
        payload = json.loads(evidence_json)
    except json.JSONDecodeError:
        return {
            "has_evidence": False,
            "summary": "Active evidence is available, but it could not be summarized.",
            "formatted_evidence": evidence_json,
        }

    document_evidence = payload.get("document_evidence") or {}
    web_evidence = payload.get("web_evidence") or {}
    document_chunks = document_evidence.get("chunks") or []
    web_results = web_evidence.get("results") or []
    route_used = payload.get("route_used") or infer_route_used(document_chunks, web_results)
    retrieval_summary = payload.get("summary") or "None"
    has_evidence = bool(document_chunks or web_results)
    summary = "\n".join(
        [
            f"Route used: {route_used}",
            f"Summary: {retrieval_summary}",
            f"Document chunk count: {len(document_chunks)}",
            f"Web result count: {len(web_results)}",
        ]
    )

    if not include_formatted_evidence or not has_evidence:
        return {
            "has_evidence": has_evidence,
            "summary": summary,
            "formatted_evidence": "",
        }

    formatted_evidence = "\n\n".join(
        section for section in [
            f"Retrieval summary:\nRoute used: {route_used}\nSummary: {retrieval_summary}",
            f"Document evidence:\n{json.dumps(document_evidence, indent=2)}" if document_chunks else "",
            f"Web evidence:\n{json.dumps(web_evidence, indent=2)}" if web_results else "",
        ]
        if section
    )

    return {
        "has_evidence": has_evidence,
        "summary": summary,
        "formatted_evidence": formatted_evidence,
    }