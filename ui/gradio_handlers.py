from __future__ import annotations

import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from queue import Empty, Queue
from shutil import copy2, rmtree
from tempfile import mkdtemp
from threading import Thread
from time import sleep
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from memory import init_memory
from orchestrator_agent import orchestrator_agent
from qdrant_vector_database import ingest_documents

DEFAULT_DOCS_DIR = ROOT_DIR / "docs"


def build_app_state(
    ready: bool = False,
    source: str = "",
    source_key: str = "docs",
) -> dict:
    return {
        "session_id": str(uuid4()),
        "ready": ready,
        "source": source,
        "source_key": source_key,
    }


def stage_uploaded_pdfs(file_paths: list[str]) -> Path:
    temp_dir = Path(mkdtemp(prefix="ragagent_pdfs_"))
    for file_path in file_paths:
        source = Path(getattr(file_path, "name", file_path))
        copy2(source, temp_dir / source.name)
    return temp_dir


def upload_signature(file_paths: list[str] | None) -> str:
    if not file_paths:
        return "docs"
    signatures = []
    for file_path in file_paths:
        path = Path(getattr(file_path, "name", file_path))
        try:
            stats = path.stat()
            signatures.append(f"{path.name}:{stats.st_size}:{stats.st_mtime_ns}")
        except OSError:
            signatures.append(path.name)
    return "|".join(sorted(signatures))


def status_text(state: dict) -> str:
    if state.get("ready"):
        return f"Research source: `{state.get('source') or 'docs/'}`"
    return "Research source: `docs/` until you upload your own PDFs to replace the current indexed documents"


def capture_logs(func, *args, **kwargs):
    buffer = StringIO()
    with redirect_stdout(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue().strip()


class LiveLogWriter:
    def __init__(self, queue: Queue):
        self.queue = queue

    def write(self, text: str) -> int:
        if text:
            self.queue.put(text)
        return len(text)

    def flush(self) -> None:
        return None


def format_trace_message(logs: str) -> str:
    logs = logs.strip()
    if not logs:
        return "Working..."
    return f"**Agent Trace**\n\n```text\n{logs}\n```"


def ingest_status_message(file_paths: list[str] | None) -> str:
    if file_paths:
        return "Ingesting uploaded documents..."
    return "Ingesting documents..."


def set_trace_entry(history: list[dict], logs: str) -> None:
    history[-1] = {"role": "assistant", "content": format_trace_message(logs)}


def append_trace_entry(history: list[dict], logs: str) -> None:
    history.append({"role": "assistant", "content": format_trace_message(logs)})


def ingest_source_documents(file_paths: list[str] | None) -> tuple[str, dict]:
    init_memory()
    staged_dir: Path | None = None

    try:
        if file_paths:
            staged_dir = stage_uploaded_pdfs(file_paths)
            pdf_dir = staged_dir
            pdf_names = sorted(Path(getattr(file_path, "name", file_path)).name for file_path in file_paths)
            source = ", ".join(pdf_names)
            source_key = upload_signature(file_paths)
        else:
            pdf_dir = DEFAULT_DOCS_DIR
            source = "docs/"
            source_key = "docs"

        info = ingest_documents(pdf_dir)
        next_state = build_app_state(ready=True, source=source, source_key=source_key)
        status = f"{status_text(next_state)} · {info['num_pdfs']} PDF(s)"
        return status, next_state
    finally:
        if staged_dir is not None:
            rmtree(staged_dir, ignore_errors=True)


def ingest_with_trace(file_paths: list[str] | None) -> tuple[str, dict, str]:
    (status, next_state), index_logs = capture_logs(ingest_source_documents, file_paths)
    logs = index_logs.strip()
    trace_message = ingest_status_message(file_paths)
    if logs:
        trace_message = f"{trace_message}\n{logs}".strip()
    return status, next_state, f"{trace_message}\nDocuments ingested.".strip()


def load_default_docs(state: dict) -> tuple[dict, str]:
    try:
        status, indexed_state, _ = ingest_with_trace(None)
        return {**indexed_state, "session_id": state["session_id"]}, status
    except Exception as exc:
        return state, f"Default docs ingestion failed: {exc}"


def clear_chat(state: dict) -> tuple[list, dict, str]:
    next_state = build_app_state(
        ready=state.get("ready", False),
        source=state.get("source", ""),
        source_key=state.get("source_key", "docs"),
    )
    return [], next_state, status_text(next_state)


def ingest_uploaded_documents(
    file_paths: list[str] | None,
    history: list[dict] | None,
    state: dict,
):
    history = history or []
    if not file_paths:
        yield history, state, status_text(state)
        return

    streamed_logs = ingest_status_message(file_paths)
    append_trace_entry(history, streamed_logs)
    yield history, {**state, "ready": False}, "Ingesting documents..."

    try:
        status, next_state, streamed_logs = ingest_with_trace(file_paths)
        set_trace_entry(history, streamed_logs)
        yield history, next_state, status
    except Exception as exc:
        history[-1] = {"role": "assistant", "content": f"Error: {exc}"}
        yield history, state, status_text(state)


def chat(message: str, history: list[dict] | None, state: dict, file_paths: list[str] | None):
    message = (message or "").strip()
    if not message:
        yield "", history, state, status_text(state)
        return

    history = history or []
    logs = ""
    streamed_logs = ""

    try:
        current_upload_key = upload_signature(file_paths) if file_paths else None
        needs_reindex = (
            not state.get("ready")
            or (
                current_upload_key is not None
                and current_upload_key != state.get("source_key")
            )
        )

        history.append({"role": "user", "content": message})

        if needs_reindex:
            streamed_logs = ingest_status_message(file_paths)
            append_trace_entry(history, streamed_logs)
            yield "", history, state, "Ingesting documents..."

            status, state, streamed_logs = ingest_with_trace(file_paths)
            set_trace_entry(history, streamed_logs)
            yield "", history, state, status
        else:
            status = status_text(state)
            append_trace_entry(history, logs)
            yield "", history, state, status

        log_queue = Queue()
        result_box: dict[str, object] = {"result": None, "error": None}

        def run_agent() -> None:
            try:
                with redirect_stdout(LiveLogWriter(log_queue)):
                    result_box["result"] = orchestrator_agent(
                        message,
                        session_id=state["session_id"],
                        verbose=True,
                    )
            except Exception as exc:
                result_box["error"] = exc
            finally:
                log_queue.put(None)

        worker = Thread(target=run_agent, daemon=True)
        worker.start()

        if not streamed_logs:
            streamed_logs = logs
        while True:
            updated = False
            while True:
                try:
                    chunk = log_queue.get_nowait()
                except Empty:
                    break

                if chunk is None:
                    if result_box["error"] is not None:
                        raise result_box["error"]

                    result = result_box["result"]
                    if result is not None:
                        if streamed_logs.strip():
                            set_trace_entry(history, streamed_logs)
                            history.append({"role": "assistant", "content": result["final_answer"]})
                        else:
                            history[-1] = {"role": "assistant", "content": result["final_answer"]}
                    yield "", history, state, status
                    return

                streamed_logs = f"{streamed_logs}{chunk}"
                updated = True

            if updated:
                set_trace_entry(history, streamed_logs)
                yield "", history, state, status

            sleep(0.1)
    except Exception as exc:
        if history and history[-1].get("role") == "assistant":
            if history[-1].get("content") != "Working...":
                history.append({"role": "assistant", "content": f"Error: {exc}"})
            else:
                history[-1] = {"role": "assistant", "content": f"Error: {exc}"}
        else:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Error: {exc}"})
        yield "", history, state, status_text(state)


INITIAL_STATE = build_app_state()
INITIAL_STATUS = status_text(INITIAL_STATE)
