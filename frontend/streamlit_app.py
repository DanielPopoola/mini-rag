import streamlit as st
import requests
import json
import time
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:9000"

class APIClient:
    """Client for communicating with FastAPI backend"""
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def check_health(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
        
    def upload_document(self, text: str, source: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Upload document via text""" 
        payload = {
            "text": text,
            "source": source,
            "title": title or source
        }

        try:
            response = requests.post(
                f"{self.base_url}/documents/upload",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Document upload failed: {e}")
            return {"error": str(e)}

    def upload_file(self, file_content: bytes, filename: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Upload document via file"""
        files = {"file": (filename, file_content)}
        data = {"title": title} if title else {}

        try:
            response = requests.post(
                f"{self.base_url}/documents/upload-file",
                files=files,
                data=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"File upload failed: {e}")
            return {"error": str(e)}

    def query(self, question: str, source_filter: Optional[str] = None) -> Dict[str, Any]:
        """Query the document collection"""
        payload = {
            "question": question,
            "source_filter": source_filter,
            "max_results": 5
        }

        try:
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Query failed: {e}")
            return {"error": str(e)}
        
    def get_stats(self) -> Dict[str, Any]:
        """Get document collection statistics"""
        try:
            response = requests.get(f"{self.base_url}/documents/stats", timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Stats request failed: {e}")
            return {"error": str(e)}
        
@st.cache_resource
def get_api_client():
    """Get cached API client instance"""
    return APIClient()

def display_citation(citation: Dict[str, Any], idx: int):
    """Display a single citation"""
    with st.container():
        st.write(f"**[{citation['citation_id']}]** {citation['title']}")
        st.caption(f"Source: {citation['source']} • Relevance: {citation['rerank_score']:.3f}")

        with st.expander(f"View citation {citation['citation_id']} text"):
            st.write(citation['text'])

def display_metrics(metrics: Dict[str, Any]):
    """Display performance metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Time", f"{metrics['total_time']:.2f}s")
    with col2:
        st.metric("Retrieval", f"{metrics['retrieval_time']:.2f}s")
    with col3:
        st.metric("Generation", f"{metrics['generation_time']:.2f}s")
    with col4:
        st.metric("Tokens", metrics['tokens_generated'])


st.set_page_config(
    page_title="Mini RAG - Frontend",
    layout="wide",
    initial_sidebar_state="expanded"
)

api_client = get_api_client()

st.title("Mini RAG")
st.caption("A lightweight RAG system that supports local models or openrouter models")

with st.container():
    health = api_client.check_health()

    if health["status"] == "healthy":
        st.success("✅ System is healthy and ready!")
    elif health["status"] == "degraded":
        st.warning("⚠️ System is running with some issues")
        with st.expander("View component status"):
            st.json(health.get("components", {}))
    else:
        st.error("❌ System is not responding. Make sure FastAPI server is running on http://localhost:9000")
        st.info("Start the server with: `uvicorn app.main:app --reload`")
        st.stop()

with st.sidebar:
    st.header("📁 Document Processing")

    upload_tab, text_tab = st.tabs(["📄 File Upload", "📝 Text Input"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["txt", "md"],
            help="Supported formats: .txt, .md"
        )

        if uploaded_file:
            file_title = st.text_input("Document title (optional)", value=uploaded_file.name)

            if st.button("Process File", type="primary"):
                with st.spinner("Processing file..."):
                    try:
                        file_content = uploaded_file.read()
                        result = api_client.upload_file(file_content, uploaded_file.name, file_title)

                        if "error" in result:
                            st.error(f"Error : {result['error']}")
                        else:
                            st.success(f"Processed {result['chunks_processed']} chunks from '{result['source']}'")
                            st.info(f"Processing time: {result['processing_time']:.2f}s")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

    with text_tab:
        text_input = st.text_area(
            "Paste your text here",
            height=200,
            placeholder="Enter document content here..."
        )

        source_name = st.text_input("Source name", placeholder="e.g., manual.txt, notes.md")
        text_title = st.text_input("Document title (optional)")

        if st.button("Process Text", type="primary", disabled=not text_input or not source_name):
            with st.spinner("Processing Text..."):
                try:
                    result = api_client.upload_document(text_input, source_name, text_title)
                    
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.success(f"✅ Processed {result['chunks_processed']} chunks from '{result['source']}'")
                        st.info(f"⏱️ Processing time: {result['processing_time']:.2f}s")
                        
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

    st.divider()

    # Collection Statistics
    st.header("Collection Stats")
    if st.button("Refresh Stats"):
        stats = api_client.get_stats()
        if "error" not in stats:
            st.metric("Documents", stats["total_documents"])
            st.metric("Vectors", stats["total_vectors"])
        else:
            st.error("Failed to load stats")

    st.divider()

    st.header("Query Settings")
    source_filter = st.text_input(
        "Filter by source (optional)",
        placeholder="e.g., manual.txt",
        help="Only search within specific document"
    )


st.header("❓ Ask Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            st.write(message["content"]["answer"])

            # Citations
            if message["content"]["citations"]:
                with st.expander(f"📚 Citations ({len(message['content']['citations'])})"):
                    for citation in message["content"]["citations"]:
                        display_citation(citation, citation["citation_id"])
            
            # Confidence indicator
            confidence = message["content"]["confidence"]
            if confidence == "high":
                st.success(f"🟢 High Confidence - {message['content']['reasoning']}")
            elif confidence == "medium":
                st.info(f"🟡 Medium Confidence - {message['content']['reasoning']}")
            elif confidence == "low":
                st.warning(f"🟠 Low Confidence - {message['content']['reasoning']}")
            else:
                st.error(f"🔴 No Answer - {message['content']['reasoning']}")
            
            # Metrics
            with st.expander("📈 Performance Metrics"):
                display_metrics(message["content"]["metrics"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = api_client.query(prompt, source_filter if source_filter else None)

            if "error" in response:
                st.error(f"❌ Query failed: {response['error']}")
                error_response = {
                    "answer": "I encountered an error while processing your question.",
                    "citations": [],
                    "confidence": "no_answer",
                    "reasoning": response['error'],
                    "metrics": {"total_time": 0, "retrieval_time": 0, "generation_time": 0, "tokens_generated": 0}
                }
                st.session_state.messages.append({"role": "assistant", "content": error_response})
            else:
                # Display successful response
                st.write(response["answer"])
                
                # Citations
                if response["citations"]:
                    with st.expander(f"📚 Citations ({len(response['citations'])})"):
                        for citation in response["citations"]:
                            display_citation(citation, citation["citation_id"])
                
                # Confidence indicator
                confidence = response["confidence"]
                if confidence == "high":
                    st.success(f"🟢 High Confidence - {response['reasoning']}")
                elif confidence == "medium":
                    st.info(f"🟡 Medium Confidence - {response['reasoning']}")
                elif confidence == "low":
                    st.warning(f"🟠 Low Confidence - {response['reasoning']}")
                else:
                    st.error(f"🔴 No Answer - {response['reasoning']}")
                
                # Metrics
                with st.expander("📈 Performance Metrics"):
                    display_metrics(response["metrics"])
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

with col2:
    if st.button("🔄 Restart Session"):
        st.session_state.clear()
        st.experimental_rerun()

with col3:
    st.caption(f"Connected to API: {API_BASE_URL}")