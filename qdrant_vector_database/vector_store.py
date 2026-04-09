import atexit
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client import models

"""
Qdrant Vector Database
=====================================================================================
Handles document ingestion and similarity search for the retriever agent.

PDF documents are loaded, titles are extracted, pages are chunked, embeddings are
created, and the chunk vectors are stored in Qdrant. The same module also exposes
similarity search and the indexed document catalog used by the retriever.
"""

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
ENV_FILE_PATH = UTILS_DIR / "var.env"
QDRANT_STORAGE_PATH = UTILS_DIR / "qdrant_storage"
INDEXED_DOCUMENTS_PATH = UTILS_DIR / "indexed_documents.json"

load_dotenv(ENV_FILE_PATH)


# get shared qdrant client for local vector storage
@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(path=str(QDRANT_STORAGE_PATH))


# close cached qdrant client on shutdown
def close_qdrant_client() -> None:
    if get_qdrant_client.cache_info().currsize == 0:
        return

    client = get_qdrant_client()
    try:
        client.close()
    except Exception:
        pass

    try:
        delattr(client, "_client")
    except AttributeError:
        pass

    get_qdrant_client.cache_clear()


atexit.register(close_qdrant_client)


# extract document title from metadata, first page, or file name
def extract_document_title(pdf_path: Path, pages: list[Any]) -> str:
    file_name_title = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
    if pages:
        metadata_title = " ".join(((pages[0].metadata or {}).get("title") or "").split()).strip()
        if metadata_title and metadata_title.casefold() != file_name_title.casefold():
            return metadata_title
        for line in pages[0].page_content.splitlines():
            candidate = " ".join(line.split()).strip()
            if len(candidate) < 12:
                continue
            if candidate.casefold().startswith(("arxiv:", "http://", "https://")):
                continue
            return candidate
    return file_name_title


# build loaded pages and saved title catalog for pdfs
def build_document_catalog(pdf_paths: list[Path]) -> tuple[list[Any], list[dict[str, str]]]:
    docs = []
    catalog = []
    for pdf_path in sorted(pdf_paths):
        pages = PyPDFLoader(str(pdf_path)).load()
        document_title = extract_document_title(pdf_path, pages)
        for page in pages:
            page.metadata["document_name"] = pdf_path.name
            page.metadata["document_title"] = document_title
        docs.extend(pages)
        catalog.append({"file_name": pdf_path.name, "title": document_title})
    return docs, catalog


# split loaded documents into chunks with citations
def chunk_documents(docs: list[Any]) -> list[Any]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        page_num = int(chunk.metadata.get("page", 0)) + 1
        document_name = chunk.metadata.get("document_name", "unknown.pdf")
        chunk.metadata.update(
            {
                "chunk_id": f"chunk_{index}",
                "page_number": page_num,
                "citation": f"[{document_name} p.{page_num}]",
            }
        )
    return chunks


# save indexed document titles for retrieval routing
def save_indexed_document_catalog(catalog: list[dict[str, str]]) -> None:
    INDEXED_DOCUMENTS_PATH.write_text(
        json.dumps(
            {
                "version": 2,
                "documents": catalog,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# get indexed document catalog from saved file or qdrant
def get_indexed_document_catalog() -> list[dict[str, str]]:
    if INDEXED_DOCUMENTS_PATH.exists():
        try:
            payload = json.loads(INDEXED_DOCUMENTS_PATH.read_text(encoding="utf-8"))
            documents = payload.get("documents", []) if payload.get("version") == 2 else []
            if documents:
                return documents
        except (OSError, json.JSONDecodeError):
            pass

    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []

    document_catalog: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=["document_name", "document_title"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            document_name = payload.get("document_name")
            if document_name:
                document_catalog.setdefault(
                    document_name,
                    payload.get("document_title") or extract_document_title(Path(document_name), []),
                )
        if offset is None:
            break

    return [
        {"file_name": document_name, "title": title}
        for document_name, title in sorted(document_catalog.items())
    ]


COLLECTION_NAME = "document_reports"
OVERSAMPLE_FACTOR = 8
MIN_FETCH_LIMIT = 20

EMBEDDING_MODELS = {
    "small": {"name": "text-embedding-3-small", "size": 1536},
    "large": {"name": "text-embedding-3-large", "size": 3072},
}
EMBEDDING_MODEL_KEY = "small"
EMBEDDING_MODEL_NAME = EMBEDDING_MODELS[EMBEDDING_MODEL_KEY]["name"]
EMBEDDING_VECTOR_SIZE = EMBEDDING_MODELS[EMBEDDING_MODEL_KEY]["size"]

embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)


# create vector embeddings and qdrant payloads for chunks
def create_document_embeddings(chunks: list[Any]) -> list[models.PointStruct]:
    texts = [chunk.page_content for chunk in chunks]
    vectors = embedding_model.embed_documents(texts)
    return [
        models.PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "content": chunk.page_content,
                "document_name": chunk.metadata.get("document_name", "unknown.pdf"),
                "document_title": chunk.metadata.get("document_title", ""),
                "page_number": chunk.metadata.get("page_number", 0),
                "chunk_id": chunk.metadata.get("chunk_id", ""),
                "citation": chunk.metadata.get("citation", ""),
                "source": chunk.metadata.get("source", ""),
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]


# reset qdrant collection before new ingestion
def reset_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )


# ingest pdf documents into qdrant and save document catalog
def ingest_documents(pdf_dir: Path) -> dict:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDFs found in {pdf_dir}")

    documents, catalog = build_document_catalog(pdf_paths)
    if not documents:
        raise ValueError(f"Unable to load PDFs from {pdf_dir}")

    document_chunks = chunk_documents(documents)
    client = get_qdrant_client()
    reset_collection(client)

    document_embeddings = create_document_embeddings(document_chunks)
    for index in range(0, len(document_embeddings), 128):
        batch = document_embeddings[index:index + 128]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,
        )

    save_indexed_document_catalog(catalog)

    return {
        "num_pdfs": len(pdf_paths),
        "num_chunks": len(document_chunks),
        "collection_name": COLLECTION_NAME,
    }


# run similarity search and return grouped chunk results
def similarity_search(
    query: str,
    per_doc_topk: int = 3,
    max_results: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []

    query_vector = embedding_model.embed_query(query)
    fetch_limit = max(max_results or 0, per_doc_topk * OVERSAMPLE_FACTOR, MIN_FETCH_LIMIT)
    points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=fetch_limit,
        with_payload=True,
    ).points

    merged: list[dict[str, Any]] = []
    counts_by_document: dict[str, int] = {}

    for point in points:
        point_score = float(point.score or 0.0)
        if score_threshold is not None and point_score < score_threshold:
            continue

        payload = point.payload or {}
        document_name = payload.get("document_name", "unknown.pdf")
        current_count = counts_by_document.get(document_name, 0)
        if current_count >= per_doc_topk:
            continue

        counts_by_document[document_name] = current_count + 1
        merged.append(
            {
                "document_name": document_name,
                "document_title": payload.get("document_title", ""),
                "page_number": payload.get("page_number", 0),
                "chunk_id": payload.get("chunk_id", ""),
                "citation": payload.get("citation", ""),
                "content": payload.get("content", ""),
                "score": point_score,
            }
        )

    merged.sort(key=lambda item: item["score"], reverse=True)
    return merged[:max_results] if max_results else merged
