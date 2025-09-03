# Mini RAG

Mini RAG is a lightweight, modular Retrieval-Augmented Generation (RAG) system designed for easy setup and experimentation. It includes a FastAPI backend for the RAG pipeline and a Streamlit frontend for user interaction. The system is configurable to use either local models via Ollama or remote models through OpenRouter.

## Architecture

The system follows a standard RAG pipeline, orchestrated by a FastAPI backend and consumed by a Streamlit UI.

```
+-----------------+      +------------------+      +-----------------+
| Streamlit UI    | <--> | FastAPI Backend  |      |   Documents     |
+-----------------+      +------------------+      +-----------------+
      |                        |                            |
      | (Query)                | (Process)                  |
      |                        v                            v
      |      +-----------------+      +---------------------+
      |      | RAG Pipeline    |      | DocumentChunker     |
      |      +-----------------+      +---------------------+
      |              |
      |              v
      |      +-----------------+      +---------------------+
      |      | RetrievalSystem |      | EmbeddingModel      |
      |      | (Vector Search) |----->| (all-MiniLM-L6-v2)  |
      |      +-----------------+      +---------------------+
      |              |                            ^
      |              v                            |
      |      +-----------------+      +---------------------+
      |      | Reranker        |      | Vector DB (Qdrant)  |
      |      | (Cross-Encoder) |<-----|                     |
      |      +-----------------+      +---------------------+
      |              |
      |              v
      |      +-----------------+
      |      | LLM Factory     |
      |      | (Selects LLM)   |
      |      +-----------------+
      |              |
      |  /-----------+-----------\
      |  v                      v
      | +--------------+     +----------------+
      | | OpenRouterLLM|     | LocalLLM(Ollama)|
      | +--------------+     +----------------+
      |        |
      | (Answer)
      v
+-----------------+
| User            |
+-----------------+
```

## Core Components & Settings

### 1. Document Chunking

- **Strategy**: `DocumentChunker` (`app/models/chunking.py`) implements a sentence-aware chunker that splits text into sentences and then groups them into chunks that respect token limits.
- **Tokenizer**: `sentence-transformers/all-MiniLM-L6-v2` (to match the embedding model).
- **Parameters**:
  - `chunk_size`: **512 tokens**
  - `overlap_size`: **50 tokens**

### 2. Embedding

- **Model**: `EmbeddingModel` (`app/models/embeddings.py`) uses `sentence-transformers/all-MiniLM-L6-v2`.
- **Dimensions**: 384

### 3. Retrieval and Reranking

- **System**: `RetrievalSystem` (`app/models/retrieval.py`) performs a two-stage retrieval process.
- **Stage 1: Vector Search**:
  - Retrieves an initial set of **20 candidates** (`initial_k=20`) from the Qdrant vector database.
- **Stage 2: Reranking**:
  - **Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2` is used to rerank the initial candidates for semantic relevance to the query.
  - The top **5 candidates** (`final_k=5`) are passed to the LLM.

### 4. LLM Generation

- **Providers**: The system can use different LLM providers, managed by a factory.
  - **`OpenRouterLLM`**: Uses the OpenRouter API. The default model is `deepseek/deepseek-chat-v3.1:free`.
  - **`LocalLLM`**: Connects to a local Ollama instance. The default model is `gemma3:1b`.
- **Factory Pattern**: The `LLMFactory` in `app/services/llm.py` demonstrates the **factory design pattern**. It provides a centralized way to create instances of different LLM providers (`OpenRouterLLM`, `LocalLLM`). This makes the system extensible—new providers can be added by creating a new class that inherits from `BaseLLM` and registering it with the factory. This decouples the RAG pipeline from the specific LLM implementation.

## Quick Start

### Prerequisites

- Git
- Python 3.9+
- Docker and Docker Compose

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/mini-rag.git
cd mini-rag

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for API keys
# (Create a .env file from the example and add your OpenRouter key)
cp .env.example .env
# Now, edit .env and add your OPENROUTER_API_KEY
```

### 2. Run with Docker (Recommended)

This is the easiest way to get started, as it handles all services (Qdrant, Ollama, Backend, Frontend).

```bash
# Build and start all services
docker-compose up --build
```

- **Streamlit Frontend**: Access at `http://localhost:8501`
- **FastAPI Backend Docs**: Access at `http://localhost:9000/docs`

### 3. Run Manually (for Development)

If you prefer to run the services outside of Docker Compose for development:

```bash
# 1. Start Qdrant and Ollama using Docker
docker-compose up -d qdrant ollama

# 2. Run the FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload

# 3. In a new terminal, run the Streamlit frontend
streamlit run frontend/streamlit_app.py
```

## Remarks

### LLM Response Parsing Tradeoffs

During development, it was observed that some LLMs, especially when instructed to generate JSON, occasionally wrap the output in markdown code blocks (e.g., ` ```json{...}``` `) or include conversational text. This caused `json.JSONDecodeError` in the backend.

To handle this, the `_parse_response` method in the `OpenRouterLLM` class was updated to robustly extract the JSON object from the raw string by locating the first `{` and the last `}`. This approach is a practical tradeoff that improves the system's resilience to inconsistent LLM behavior, preventing crashes and allowing the application to proceed even with imperfectly formatted responses.

### Provider Flexibility

The use of the **Factory Pattern** in `app/services/llm.py` is a key architectural feature. It allows developers to easily switch between LLM providers (e.g., from `openrouter` to `ollama`) by changing a single configuration parameter. It also makes the system highly extensible: adding a new provider like Anthropic or Cohere would only require creating a new class that implements the `BaseLLM` interface and registering it with the `LLMFactory`.