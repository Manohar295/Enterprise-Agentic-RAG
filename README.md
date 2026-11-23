# 📚 Enterprise-Grade Agentic RAG Assistant

An AI-powered document question-answering application built using LangChain, LangGraph, Groq, Hugging Face and FAISS.

## 🌐 Live Demo

🚀 [Click Here to Try the Application](PASTE_Yhttps://enterprise-agentic-rag-zkwy97pvhncrm4fdjzdxtv.streamlit.app/OUR_STREAMLIT_URL_HERE)

## ✨ Features

- Upload PDF, DOCX, PPTX and TXT documents.
- Agentic RAG workflow using LangGraph.
- Semantic search using FAISS Vector Database.
- Hugging Face Embeddings.
- Groq LLM integration.
- Context-based AI responses with source references.

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- LangGraph
- Groq LLM
- Hugging Face
- FAISS

## 💻 Run Locally

1. Create or activate a Python 3.11 environment.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Create `.streamlit/secrets.toml` and add your Groq API key:

   `GROQ_API_KEY = "your_groq_api_key"`

4. Start the application:

   `streamlit run app.py`

## ☁️ Deployment

Deployed using Streamlit Community Cloud.

## 📝 Note

The application indexes uploaded documents during the running session. Upload and build the knowledge base again after a new app session or restart.
