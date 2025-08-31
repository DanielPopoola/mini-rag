import streamlit as st
import sys
import os
from pathlib import Path
import time
import logging

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.models.rag_pipeline import RAGPipeline
from app.models.embeddings import EmbeddingModel
from app.models.chunking import DocumentChunker
from app.models.retrieval import RetrievalSystem
from app.services.vector_db import VectorDatabase
from app.services.llm import LocalLLM, OpenRouterLLM

logger = logging.getLogger(__name__)

# --- App Configuration ---
st.set_page_config(
    page_title="Mini RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DI Container ---
@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()

@st.cache_resource
def get_vector_db():
    vector_db = VectorDatabase()
    vector_db.create_collection(dimension=384)
    return vector_db

@st.cache_resource
def get_llm():
    return OpenRouterLLM(model_name="deepseek/deepseek-chat-v3.1:free")

@st.cache_resource
def get_rag_pipeline():
    embedding_model = get_embedding_model()
    vector_db = get_vector_db()
    llm = get_llm()
    
    chunker = DocumentChunker()
    retrieval_system = RetrievalSystem(embedding_model, vector_db)
    
    return RAGPipeline(
        embedding_model=embedding_model,
        vector_db=vector_db,
        retrieval_system=retrieval_system,
        llm=llm,
        chunker=chunker
    )


st.title("Mini RAG")
st.caption("A mini RAG application powered by local models")


with st.sidebar:
    st.header("📁 Document Processing")
    uploaded_file = st.file_uploader(
        "Upload a document (.txt)",
        type=["txt", "pdf", "docx", "md"]
    )

    def extract_text(uploaded_file):
        """Extract text from different file types"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == "txt" or file_extension == "md":
            return uploaded_file.read().decode("utf-8")
        elif file_extension == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            return "".join(page.extract_text() for page in reader.pages)
        elif file_extension == "docx":
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
            
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                rag_pipeline = get_rag_pipeline()
                
                try:
                    content = extract_text(uploaded_file)
                    if content:
                        metadata = {"source": uploaded_file.name}
                        
                        result = rag_pipeline.process_document(content, metadata)
                        
                        st.success(f"Processed {result['chunks_processed']} chunks from {result['source']}")
                        st.info(f"Processing time: {result['processing_time']:.2f}s")
                    
                except Exception as e:
                    st.error(f"Failed to process document: {e}")

    st.divider()
    st.header("⚙️ Settings")
    if st.button("Show Collection Info"):
        try:
            vector_db = get_vector_db()
            info = vector_db.get_collection_info()
            st.json(info)
        except Exception as e:
            st.error(f"Error getting collection info: {e}")
    source_filter = st.text_input("Filter by source (optional)")
    rerank_threshold = st.slider(
        "Rerank Threshold", 
        min_value=-10.0, 
        max_value=10.0, 
        value=0.0, 
        step=0.1,
        help="Minimum score for a retrieved chunk to be considered relevant."
    )



st.header("❓ Ask a Question")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What do you want to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):
            try:
                rag_pipeline = get_rag_pipeline()
                vector_info = rag_pipeline.vector_db.get_collection_info()

                if vector_info["points_count"] == 0:
                    st.info("👋 Welcome! Upload a document first to get started.")
            except Exception as e:
                st.error("⚠️ System initialization failed. Check if Ollama is running.")
                st.stop()
            
            try:
                response = rag_pipeline.query(prompt, source_filter if source_filter else None, rerank_threshold=rerank_threshold)

                if response["confidence"] == "no_answer":
                    st.warning("⚠️ I don't have enough information to answer this question.\
                     Try uploading relevant documents first.")
                elif response["confidence"] == "low":
                    st.info("💡 My confidence in this answer is low. The information might be incomplete.")
                
                # Display answer
                st.markdown(response["answer"])
                
                # Display citations
                if response["citations"]:
                    with st.expander("📚 Citations"):
                        for citation in response["citations"]:
                            st.write(f"**Source:** {citation['source']} (Score: {citation['rerank_score']:.2f})")
                            st.caption(citation["text"])
                
                # Display metrics
                with st.expander("📊 Metrics"):
                    st.metric("Total Time", f"{response['metrics']['total_time']:.2f}s")
                    st.metric("Retrieval Time", f"{response['metrics']['retrieval_time']:.2f}s")
                    st.metric("Generation Time", f"{response['metrics']['generation_time']:.2f}s")
                    st.metric("Tokens Generated", response['metrics']['tokens_generated'])
                    st.metric("Retrieved Chunks", response['retrieved_chunks'])
                
                full_response = response["answer"]

            except ConnectionError:
                st.error("🔌 Cannot connect to Ollama. Make sure it's running: `ollama serve`")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Query error: {e}", exc_info=True)
                full_response = "Sorry, I encountered an error."

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
