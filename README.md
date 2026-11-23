# Enterprise-Grade Agentic RAG — Streamlit

## Run locally
1. Create/activate a Python 3.11 environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.streamlit/secrets.toml` with:
   `GROQ_API_KEY = "your_groq_api_key"`
4. Start: `streamlit run app.py`

## Deploy to Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. On https://share.streamlit.io, choose the repository and `app.py`.
3. In App settings → Secrets, add `GROQ_API_KEY = "your_groq_api_key"`.
4. Deploy.

The app indexes uploaded files in the running session. Upload and build the knowledge base again after a new app session/restart.
