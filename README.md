# Mini RAG

This project is a simple Retrieval-Augmented Generation (RAG) application that uses a local vector database and a large language model to answer questions based on a provided document.

## Architecture

The application is composed of a frontend and a backend.

### Frontend

The frontend is a Streamlit application that provides a user interface for interacting with the RAG pipeline. It allows users to upload a document, ask questions, and view the answers.

### Backend

The backend is a FastAPI application that exposes an API for the RAG pipeline. The frontend communicates with the backend to process documents and generate answers.

The backend consists of the following components:

*   **RAG Pipeline:** The core of the application, responsible for orchestrating the entire process of document chunking, embedding, retrieval, and answer generation.
*   **Chunking:** The document is split into smaller chunks to facilitate efficient embedding and retrieval.
*   **Embeddings:** Each chunk is converted into a vector embedding using a sentence transformer model.
*   **Vector DB:** The embeddings are stored in a local vector database (ChromaDB) for efficient similarity search.
*   **Retrieval:** When a question is asked, the most relevant chunks are retrieved from the vector database based on the question's embedding.
*   **LLM:** The retrieved chunks and the original question are passed to a large language model (Google's Gemini) to generate a final answer.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the backend:**
    ```bash
    uvicorn app.main:app --reload
    ```
3.  **Run the frontend:**
    ```bash
    streamlit run frontend/streamlit_app.py
    ```

## Key Components

*   **`app/main.py`:** The main entry point for the FastAPI backend.
*   **`app/models/rag_pipeline.py`:** Orchestrates the RAG pipeline.
*   **`app/models/chunking.py`:** Handles document chunking.
*   **`app/models/embeddings.py`:** Generates embeddings for the chunks.
*   **`app/services/vector_db.py`:** Manages the ChromaDB vector database.
*   **`app/models/retrieval.py`:** Retrieves relevant chunks from the vector database.
*   **`app/services/llm.py`:** Interacts with the Google Gemini API.
*   **`frontend/streamlit_app.py`:** The Streamlit frontend application.
