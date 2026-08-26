import os, re, json
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Agentic RAG Schedule Assistant", page_icon="📅")
BASE_DATE = date(2026, 8, 26)
DATA_FILE = Path(__file__).parent / "schedule.json"

def load_events():
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))

@st.cache_resource
def resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(Path(__file__).parent / "chroma_db"))
    collection = client.get_or_create_collection("schedule")
    return model, collection

embedding_model, collection = resources()

def event_text(e):
    return f"Title: {e['title']}. Type: {e['type']}. Date: {e['date']}. Time: {e['start']} to {e['end']}. Location: {e['location']}. Notes: {e['notes']}."

def sync_chroma():
    data = load_events()
    ids = {e["id"] for e in data}
    existing = collection.get().get("ids", [])
    stale = [x for x in existing if x not in ids]
    if stale:
        collection.delete(ids=stale)
    if data:
        texts = [event_text(e) for e in data]
        collection.upsert(
            ids=[e["id"] for e in data],
            documents=texts,
            metadatas=data,
            embeddings=embedding_model.encode(texts).tolist()
        )

sync_chroma()

def parse_date(text):
    t = text.lower()
    if "day after tomorrow" in t:
        return BASE_DATE + timedelta(days=2)
    if "tomorrow" in t:
        return BASE_DATE + timedelta(days=1)
    if "today" in t:
        return BASE_DATE
    weekdays = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
    for name, idx in weekdays.items():
        if name in t:
            return BASE_DATE + timedelta(days=(idx - BASE_DATE.weekday()) % 7)
    m = re.search(r"\b(august|september)\s+(\d{1,2})\b", t)
    if m:
        return date(2026, 8 if m.group(1) == "august" else 9, int(m.group(2)))
    m = re.search(r"\b(2026-\d{2}-\d{2})\b", t)
    return date.fromisoformat(m.group(1)) if m else None

def normalize_time(value):
    if not value:
        return ""
    m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", str(value).lower())
    if not m:
        return ""
    h, minute, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and h < 12: h += 12
    if ap == "am" and h == 12: h = 0
    return f"{h:02d}:{minute:02d}" if h <= 23 and minute <= 59 else ""

@tool
def get_schedule(query: str, target_date: str = "", start_time: str = "", end_time: str = "") -> str:
    """Retrieve schedule information. Use this for existing events, availability, conflicts, dates, times, and finding events."""
    d = target_date or (parse_date(query).isoformat() if parse_date(query) else "")
    q = query or f"schedule {d}"
    results = collection.query(
        query_embeddings=embedding_model.encode([q]).tolist(),
        n_results=min(10, max(1, collection.count()))
    )
    ids = results.get("ids", [[]])[0]
    data = load_events()
    candidates = [e for e in data if (not d or e["date"] == d)]
    if not d:
        candidates = [e for e in data if e["id"] in ids]
    start, end = normalize_time(start_time), normalize_time(end_time)
    if start: candidates = [e for e in candidates if e["start"] >= start]
    if end: candidates = [e for e in candidates if e["start"] < end]
    return json.dumps(candidates, indent=2)

@tool
def update_schedule(action: str, event_id: str = "", title: str = "", event_type: str = "meeting",
                    event_date: str = "", start_time: str = "", end_time: str = "",
                    location: str = "", notes: str = "") -> str:
    """Add, update, or remove schedule entries. action must be add, update, or remove."""
    data = load_events()
    if action == "add":
        if not title or not event_date or not start_time:
            return "ERROR: title, event_date and start_time are required."
        new_id = "evt" + str(max([int(e["id"][3:]) for e in data] + [0]) + 1).zfill(3)
        start = normalize_time(start_time)
        end = normalize_time(end_time) or f"{(int(start[:2])+1):02d}:{start[3:]}"
        event = {"id":new_id,"title":title,"type":event_type,"date":event_date,
                 "start":start,"end":end,"location":location,"notes":notes}
        data.append(event)
        DATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        sync_chroma()
        return json.dumps({"status":"added","event":event})
    if action == "remove":
        new_data = [e for e in data if e["id"] != event_id]
        if len(new_data) == len(data): return "ERROR: event not found."
        DATA_FILE.write_text(json.dumps(new_data, indent=2), encoding="utf-8")
        sync_chroma()
        return json.dumps({"status":"removed","event_id":event_id})
    if action == "update":
        for e in data:
            if e["id"] == event_id:
                if title: e["title"] = title
                if event_date: e["date"] = event_date
                if start_time: e["start"] = normalize_time(start_time)
                if end_time: e["end"] = normalize_time(end_time)
                if event_type: e["type"] = event_type
                if location: e["location"] = location
                if notes: e["notes"] = notes
                DATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
                sync_chroma()
                return json.dumps({"status":"updated","event":e})
        return "ERROR: event not found."
    return "ERROR: action must be add, update, or remove."

TOOLS = [get_schedule, update_schedule]
TOOL_MAP = {t.name:t for t in TOOLS}

def get_llm():
    key = os.getenv("GOOGLE_API_KEY", "")
    try:
        if not key:
            key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    if not key:
        return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key, temperature=0)

def agent(user_query):
    llm = get_llm()
    if llm is None:
        return "⚠️ GOOGLE_API_KEY is not configured. Add it to Streamlit Secrets."
    system = SystemMessage(content=f"""
You are an Agentic RAG Schedule Assistant. Today is {BASE_DATE.isoformat()}.
You have exactly two tools:
1. get_schedule: retrieve existing schedule information using ChromaDB RAG.
2. update_schedule: add, update, or remove schedule entries.

Rules:
- Never guess schedule data. Use get_schedule for schedule questions.
- For availability questions, retrieve the requested date and time range and determine whether it is free.
- For add/change/delete requests, use update_schedule.
- For moving an event, first use get_schedule to identify the event, then update_schedule.
- Resolve relative dates from today's date.
- After tool execution, provide a concise natural-language answer.
""")
    messages = [system, HumanMessage(content=user_query)]
    model = llm.bind_tools(TOOLS)
    for _ in range(5):
        ai = model.invoke(messages)
        messages.append(ai)
        calls = getattr(ai, "tool_calls", [])
        if not calls:
            return ai.content
        for call in calls:
            result = TOOL_MAP[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return "I could not complete the request."

st.title("📅 Agentic RAG Schedule Assistant")
st.caption("Gemini + LangChain tool calling + ChromaDB vector database")

with st.sidebar:
    st.subheader("System")
    st.write("**LLM:** Gemini 2.5 Flash")
    st.write("**Agent framework:** LangChain")
    st.write("**Vector DB:** ChromaDB")
    st.write("**Embeddings:** all-MiniLM-L6-v2")
    st.write("**Tools:** get_schedule, update_schedule")
    st.divider()
    st.write(f"Events: {len(load_events())}")
    if get_llm() is None:
    st.warning("GOOGLE_API_KEY is not configured.")
    else:
    st.success("Gemini API connected.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about your schedule or make a change...")
if query:
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Agent is working..."):
            answer = agent(query)
        st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})
