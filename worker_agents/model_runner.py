import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
ENV_FILE_PATH = UTILS_DIR / "var.env"

load_dotenv(ENV_FILE_PATH)

@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

""" helper function to be used by the agents to run the LLM """
def run_model(
        *,
        instructions: str,
        input_data: Any,
        tools: Optional[List[Dict[str, Any]]] = None,
        previous_response_id: Optional[str] = None,
        model: str = None,
        reasoning_effort: Optional[str] = None,
        text_format: Any = None,
):
    client = get_openai_client()
    request = dict(
        model=model,
        instructions=instructions,
        input=input_data,
        tools=tools or [],
        tool_choice="auto",
        previous_response_id=previous_response_id,
        parallel_tool_calls=False,
    )
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}
    if text_format is not None:
        return client.responses.parse(**request, text_format=text_format)
    return client.responses.create(**request)