from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_vector_database import close_qdrant_client
from ui.gradio_handlers import (
    INITIAL_STATE,
    INITIAL_STATUS,
    chat,
    clear_chat,
    ingest_uploaded_documents,
    load_default_docs,
)

CHATBOT_ELEM_ID = "chatbot"
CHATBOT_SCROLL_HEAD = f"""
<script>
(() => {{
  const chatbotId = "{CHATBOT_ELEM_ID}";
  let mounted = false;

  function mountFinalAnswerScrollFix() {{
    const root = document.getElementById(chatbotId);
    const log = root?.querySelector('[role="log"]');
    if (!log || mounted) {{
      return;
    }}

    mounted = true;
    let lastBotCount = 0;
    let lastBotText = "";

    const observer = new MutationObserver(() => {{
      const botRows = Array.from(log.querySelectorAll(".message-row.bot-row"));
      const lastBot = botRows.at(-1);
      const currentText = (lastBot?.innerText || "").trim();
      const isTraceMessage = currentText.includes("Agent Trace");
      const shouldResetView =
        lastBot &&
        !isTraceMessage &&
        currentText &&
        currentText != lastBotText &&
        (botRows.length > lastBotCount || lastBotText.includes("Agent Trace"));

      if (shouldResetView) {{
        requestAnimationFrame(() => {{
          lastBot.scrollIntoView({{ block: "start", behavior: "auto" }});
        }});
      }}

      lastBotCount = botRows.length;
      lastBotText = currentText;
    }});

    observer.observe(log, {{
      childList: true,
      subtree: true,
      characterData: true,
    }});
  }}

  const rootObserver = new MutationObserver(mountFinalAnswerScrollFix);
  rootObserver.observe(document.body, {{ childList: true, subtree: true }});

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", mountFinalAnswerScrollFix, {{ once: true }});
  }} else {{
    mountFinalAnswerScrollFix();
  }}
}})();
</script>
"""
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(CHATBOT_SCROLL_HEAD)
    gr.Markdown(
        "## Multi-Agent Research RAG\n"
        "Researches from the active documents and the internet when needed. "
        "Upload your own PDFs if you want to replace the current indexed documents."
    )

    app_state = gr.State(INITIAL_STATE)
    status = gr.Markdown(INITIAL_STATUS)

    with gr.Accordion("Replace Documents", open=False):
        uploads = gr.File(
            label="Upload Your PDFs",
            file_count="multiple",
            file_types=[".pdf"],
            type="filepath",
        )

    with gr.Row():
        clear_button = gr.Button("Clear chat")

    chatbot = gr.Chatbot(
        show_label=False,
        elem_id=CHATBOT_ELEM_ID,
        height=540,
    )
    message = gr.Textbox(
        show_label=False,
        placeholder="Research on a topic",
        container=False,
    )

    demo.load(load_default_docs, inputs=app_state, outputs=[app_state, status])
    clear_button.click(clear_chat, inputs=app_state, outputs=[chatbot, app_state, status])
    uploads.change(
        ingest_uploaded_documents,
        inputs=[uploads, chatbot, app_state],
        outputs=[chatbot, app_state, status],
    )
    message.submit(chat, inputs=[message, chatbot, app_state, uploads], outputs=[message, chatbot, app_state, status])

demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    try:
        demo.launch()
    finally:
        close_qdrant_client()
