import os
import tempfile
from pathlib import Path
from typing import TypedDict, List, Dict, Any

import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END


st.set_page_config(page_title="Enterprise Agentic RAG", page_icon="📚", layout="wide")
st.title("📚 Enterprise-Grade Agentic RAG Assistant")
st.caption("Upload your documents and ask questions grounded in their content.")

# ---------- Secrets / model ----------
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
if not api_key:
    st.error("Groq API key not found. Add GROQ_API_KEY in Streamlit Secrets.")
    st.info('In Streamlit Cloud: App → Settings → Secrets, add: GROQ_API_KEY = "your_key"')
    st.stop()

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
def load_llm(key: str):
    return ChatGroq(
        api_key=key,
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=2048,
    )

llm = load_llm(api_key)
embedding_model = load_embedding_model()


# ---------- Document extraction ----------
def extract_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"\n[Page {page_number}]\n{page_text}")
        return "\n".join(pages)

    if suffix == ".docx":
        doc = DocxDocument(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)

    if suffix == ".pptx":
        prs = Presentation(str(path))
        slides_text = []
        for slide_number, slide in enumerate(prs.slides, start=1):
            items = [
                shape.text.strip()
                for shape in slide.shapes
                if hasattr(shape, "text") and shape.text.strip()
            ]
            if items:
                slides_text.append(f"[Slide {slide_number}]\n" + "\n".join(items))
        return "\n\n".join(slides_text)

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    return ""


# ---------- Agentic RAG graph ----------
class RAGState(TypedDict, total=False):
    question: str
    chat_history: str
    search_query: str
    documents: List[Document]
    relevance: str
    answer: str
    grounded: str
    retry_count: int
    sources: List[str]


def build_agent(retriever):
    def rewrite_question(state: RAGState) -> RAGState:
        # Keep the user's question unchanged; the retrieval retry can improve it.
        return {"search_query": state["question"]}

    def retrieve_documents(state: RAGState) -> RAGState:
        docs = retriever.invoke(state.get("search_query") or state["question"])
        return {"documents": docs}

    def grade_relevance(state: RAGState) -> RAGState:
        return {"relevance": "yes" if state.get("documents") else "no"}

    def generate_answer(state: RAGState) -> RAGState:
        docs = state.get("documents", [])
        context_parts, source_names = [], []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            source_names.append(source)
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")

        context = "\n\n".join(context_parts) or "No relevant document excerpts were retrieved."
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an enterprise knowledge assistant. Answer using ONLY the supplied context. "
             "If the context does not contain the answer, clearly say: "
             "'I could not find this information in the uploaded documents.' "
             "Do not invent facts. Cite supporting excerpts inline using [Source 1], [Source 2], etc. "
             "Keep the answer clear and structured.\n\nCONTEXT:\n{context}"),
            ("human", "Conversation history:\n{history}\n\nQuestion: {question}")
        ])
        result = (prompt | llm).invoke({
            "context": context,
            "history": state.get("chat_history", ""),
            "question": state["question"],
        })
        return {
            "answer": result.content.strip(),
            "sources": list(dict.fromkeys(source_names)),
        }

    def verify_answer(state: RAGState) -> RAGState:
        docs = state.get("documents", [])
        if not docs:
            return {"grounded": "no"}
        context = "\n\n".join(d.page_content[:1500] for d in docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Check whether the answer is supported by the provided context. "
             "Return exactly GROUNDED if key factual claims are supported, otherwise NOT_GROUNDED."),
            ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
        ])
        result = (prompt | llm).invoke({
            "context": context,
            "answer": state.get("answer", ""),
        })
        verdict = result.content.strip().upper()
        return {"grounded": "yes" if verdict.startswith("GROUNDED") else "no"}

    def retry_or_finish(state: RAGState) -> str:
        if state.get("retry_count", 0) < 1 and (
            state.get("relevance") == "no" or state.get("grounded") == "no"
        ):
            return "retry"
        return "finish"

    def improve_search_query(state: RAGState) -> RAGState:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Create a better, more specific search query for retrieving evidence from uploaded documents. "
             "Return only the query."),
            ("human",
             "Original question: {question}\nPrevious query: {query}\n"
             "The retrieved evidence was insufficient or the draft was not grounded. "
             "Write an improved query.")
        ])
        result = (prompt | llm).invoke({
            "question": state["question"],
            "query": state.get("search_query", state["question"]),
        })
        return {
            "search_query": result.content.strip(),
            "retry_count": state.get("retry_count", 0) + 1,
        }

    workflow = StateGraph(RAGState)
    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_relevance)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("verify", verify_answer)
    workflow.add_node("improve_query", improve_search_query)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        lambda s: "generate" if s.get("relevance") == "yes" else (
            "retry" if s.get("retry_count", 0) < 1 else "generate"
        ),
        {"generate": "generate", "retry": "improve_query"},
    )
    workflow.add_edge("improve_query", "retrieve")
    workflow.add_edge("generate", "verify")
    workflow.add_conditional_edges(
        "verify", retry_or_finish,
        {"retry": "improve_query", "finish": END},
    )
    return workflow.compile()


# ---------- Sidebar: upload and index ----------
with st.sidebar:
    st.header("1. Upload knowledge base")
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "pptx", "txt"],
        accept_multiple_files=True,
    )
    build_clicked = st.button("Build / Refresh knowledge base", type="primary", use_container_width=True)
    st.caption("Supported formats: PDF, DOCX, PPTX, TXT")

if build_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Reading documents, creating embeddings and building FAISS index..."):
            raw_documents = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded in uploaded_files:
                    safe_name = Path(uploaded.name).name
                    file_path = Path(temp_dir) / safe_name
                    file_path.write_bytes(uploaded.getvalue())
                    extracted = extract_file(file_path)
                    if extracted.strip():
                        raw_documents.append(
                            Document(page_content=extracted, metadata={"source": safe_name})
                        )

            if not raw_documents:
                st.error("No readable text found. Please check that your files contain selectable text.")
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                chunks = splitter.split_documents(raw_documents)
                for idx, chunk in enumerate(chunks):
                    chunk.metadata["chunk_id"] = idx

                vector_store = FAISS.from_documents(chunks, embedding_model)
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5},
                )
                st.session_state.vector_store = vector_store
                st.session_state.agent = build_agent(retriever)
                st.session_state.indexed_files = [d.metadata["source"] for d in raw_documents]
                st.session_state.messages = []
                st.success(f"Knowledge base ready: {len(raw_documents)} document(s), {len(chunks)} chunks.")

if "agent" not in st.session_state:
    st.info("Upload one or more documents in the sidebar, then click **Build / Refresh knowledge base** to start.")
    st.stop()

with st.sidebar:
    st.success("Knowledge base is ready")
    for filename in st.session_state.get("indexed_files", []):
        st.write("📄", filename)
    if st.button("Clear chat"):
        st.session_state.messages = []


# ---------- Chat interface ----------
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

question = st.chat_input("Ask a question about your uploaded documents...")
if question:
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in st.session_state.get("messages", [])[-6:]
    )
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent is retrieving, checking and preparing an answer..."):
            try:
                result = st.session_state.agent.invoke({
                    "question": question.strip(),
                    "chat_history": history_text,
                    "retry_count": 0,
                })
                answer = result.get("answer", "I could not generate an answer.")
                sources = result.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(f"• {source}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as exc:
                st.error(f"An error occurred: {exc}")
