# Multi Agent RAG Researcher

Multi Agent RAG Researcher is based on an orchestrator that coordinates three worker agents to work together on a topic and generate grounded content. It works with two evidence sources: local PDF documents and the web.

The three worker agents are:

- Retriever Agent: retrieves information from local documents (PDFs), the web, or both.
- Writer Agent: writes the response based on the retrieved evidence.
- Verifier Agent: checks the draft written by the Writer Agent and returns the final verified response.

## Key Components

1. `gpt-5.4` model used by the orchestrator and all worker agents
2. Step-by-step function calling that allows the agents to interact with one another
3. Qdrant vector database for local PDF retrieval
4. Tavily for web search
5. SQLite for short-term memory
6. Gradio UI for browser-based interaction

## Multi-Agent Architecture

### Data Sources

1. Qdrant Vector Database

Information retrieval from PDFs is handled in the following stages:

- Multiple PDFs can be loaded from the `docs/` folder or uploaded through the UI.
- Documents are split into chunks, converted into embeddings, and stored in a local Qdrant collection.
- Similarity search is then used to retrieve the most relevant chunks across the indexed documents.
- The retrieved chunks include citation metadata such as document name and page number.

2. Tavily Web Search

Tavily is used to retrieve up-to-date or external information from the web. The retriever can use it when:

- the indexed PDFs do not cover the query
- document evidence is weak or incomplete
- newer information is needed

### Worker Agents

1. Retriever Agent

The role is:

- It uses two tools: PDF document retrieval and web search.
- Given a query, it decides whether to use local documents, web search, or both.
- If local document evidence is missing or weak, it can fall back to web search to gather broader or more up-to-date context.

2. Writer Agent

The role is:

- It receives the retrieved information from the Retriever Agent.
- It writes a grounded draft based on the available evidence.
- It includes supporting citations from PDFs or web sources when they are available.

3. Verifier Agent

The role is:

- It receives the draft from the Writer Agent together with the evidence.
- It checks whether the claims in the draft are supported by the retrieved evidence.
- It returns the final verified response.

### Memory

SQLite is used to provide short-term memory for the multi-agent workflow. For a given session ID, the system stores:

- the latest user query
- the latest retrieved evidence for that session

This allows the orchestrator to reuse relevant evidence for follow-up questions instead of retrieving the same information again every time.

### Orchestrator

The orchestrator coordinates the three worker agents: Retriever, Writer, and Verifier.

#### Working Mechanism

- It receives the user query and, depending on the query, may respond directly or begin the evidence-based workflow.
- For a research query, it first checks whether relevant cached evidence from the current session can be reused.
- If cached evidence is not enough, it calls the Retriever Agent to gather evidence from PDFs, the web, or both.
- If there is document evidence but the evidence is weak, the Retriever Agent can also fetch up-to-date information from the web to supplement the local document information.
- The orchestrator then passes the active evidence and the user query to the Writer Agent so it can generate a grounded draft.
- Next, it sends the draft and evidence to the Verifier Agent, which checks the claims and returns the final verified report.
- During the session, the latest query and retrieved evidence are stored in memory for follow-up questions.
- In follow-up questions, the orchestrator may reuse cached evidence instead of calling the Retriever Agent again, then continue with the Writer Agent and Verifier Agent to generate the final response.

Note: The orchestrator has a guardrail that keeps the system focused on research and factual questions. It refuses unrelated general tasks such as coding help or simple math because the goal of the system is to function as a research assistant.

## Project Structure

```text
.
├── docs/                         # Default PDF files
├── memory/                       # SQLite-backed session memory helpers
├── qdrant_vector_database/       # PDF ingestion and similarity search
├── ui/                           # Gradio app and UI handlers
├── utils/
│   ├── requirements.txt          # Python dependencies
│   ├── var.env                   # Local API keys
│   ├── memory.db                 # Created at runtime
│   └── qdrant_storage/           # Created at runtime
├── worker_agents/                # Retriever, writer, and verifier
├── orchestrator_agent.py         # Main coordinator
└── run_orchestrator.py           # CLI entry point
```

## Setup Project

### Prerequisites

- Python 3.10 or newer
- OpenAI API key
- Tavily API key

### Installation

1. Clone the repository:

```bash
git clone https://github.com/ayoolaolafenwa/multi-agent-rag-researcher.git
cd multi-agent-rag-researcher
```

2. Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

3. Install the dependencies:

```bash
pip3 install -r utils/requirements.txt
pip3 install langchain-openai
```

4. Create a `utils/var.env` file and store your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

5. Place the PDFs you want to index in the `docs/` folder, or upload PDFs later through the UI. The project already includes existing PDFs in `docs/`, currently `Gemma 3 Technical Report.pdf` and `DeepSeek-V3.2.pdf`, so you can use those directly or replace them with your own documents.

## Run Project

Start the command-line app:

```bash
python3 run_orchestrator.py
```

When the CLI starts, it ingests the PDFs in `docs/` into the local Qdrant store. Type `q` or `exit` to end the session.

## Run UI for Multi-Agent Chat

Start the Gradio UI:

```bash
python3 ui/gradio_app.py
```

The UI automatically loads the default PDFs from `docs/` on startup. If you upload new PDFs, they replace the active indexed document set for that UI session.

## Notes

- Session memory is stored in `utils/memory.db`.
- Local Qdrant data is stored in `utils/qdrant_storage/`.
- The system is designed for research and factual question answering, not for unrelated general-purpose tasks.
