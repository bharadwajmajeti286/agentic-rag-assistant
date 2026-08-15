# Agentic RAG Schedule Assistant

A Streamlit-based Agentic RAG schedule assistant for a 30-day schedule.

## Features
- Sample schedule covering 30 days
- ChromaDB vector database
- Sentence Transformer embeddings (`all-MiniLM-L6-v2`)
- Agent/router that chooses retrieval or schedule modification
- `get_schedule` tool for semantic/date/time retrieval
- `update_schedule` tool for add/update/remove
- Streamlit user interface

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `schedule.json`, `requirements.txt`, and `README.md`.
3. In Streamlit Community Cloud, create a new app.
4. Select the repository, branch `main`, and entrypoint `app.py`.
5. Deploy and copy the generated public URL into `deployed_url.txt`.
