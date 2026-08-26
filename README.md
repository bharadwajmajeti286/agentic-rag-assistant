# Agentic RAG Schedule Assistant

## Technology stack
- Streamlit UI
- Google Gemini 2.5 Flash LLM
- LangChain tool calling
- ChromaDB vector database
- Sentence Transformers: all-MiniLM-L6-v2
- JSON schedule store

## Required tools
- `get_schedule` — RAG retrieval from ChromaDB with date/time filters.
- `update_schedule` — add, update, and remove schedule entries; ChromaDB is synchronized after every mutation.

## Run locally
```bash
pip install -r requirements.txt
```

Set your Gemini key as an environment variable:
```powershell
$env:GOOGLE_API_KEY="YOUR_KEY"
streamlit run app.py
```

## Streamlit Cloud
Deploy `app.py`. In App Settings → Secrets, add:
```toml
GOOGLE_API_KEY = "YOUR_KEY"
```

Do NOT commit the API key or `.streamlit/secrets.toml`.

## Test
- What do I have scheduled tomorrow?
- Am I free Friday afternoon?
- Add a meeting on August 28 at 3 PM.
- Move my meeting from 2 PM to 4 PM.
- Delete my AI workshop.
