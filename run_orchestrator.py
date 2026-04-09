from __future__ import annotations

from uuid import uuid4
from pathlib import Path

from memory import init_memory
from orchestrator_agent import orchestrator_agent
from qdrant_vector_database import close_qdrant_client, ingest_documents

def initialize_app(pdf_dir: Path) -> None:
    init_memory()
    print("Ingesting documents...")
    info = ingest_documents(pdf_dir)
    print(
        f"{info['num_pdfs']} PDFs ingested to Qdrant collection {info['collection_name']}"
    )

def chat_with_supervisor(session_id: str | None = None) -> None:
    if session_id is None:
        session_id = str(uuid4())

    print("Analyze your pdfs!! \n")
    print("Use 'q', 'exit', or 'exist' to end chat. \n")
    while True:
        user_query = input("User: ").strip()

        if not user_query:
            continue

        if user_query.lower() in {"q", "exit", "exist"}:
            print("Exiting chat loop.")
            break

        result = orchestrator_agent(user_query, session_id=session_id, verbose=True)
        print("Assistant:", result["final_answer"])
        print()

if __name__ == "__main__":
    pdf_dir = Path("docs")
    try:
        initialize_app(pdf_dir)
        chat_with_supervisor()
    finally:
        close_qdrant_client()
