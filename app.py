import os
import re
import json
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Agentic RAG Schedule Assistant", page_icon="📅", layout="centered")

BASE_DATE = date(2026, 8, 26)
DATA_FILE = Path(__file__).parent / "schedule.json"


def load_events():
    if not DATA_FILE.exists():
        DATA_FILE.write_text("[]", encoding="utf-8")
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def save_events(events):
    DATA_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")


@st.cache_resource
def get_vector_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=str(Path(__file__).parent / "chroma_db"))
    collection = chroma_client.get_or_create_collection(name="schedule")
    return embedding_model, collection


embedding_model, collection = get_vector_resources()


def event_to_text(event):
    return (
        f"Title: {event['title']}. Type: {event['type']}. Date: {event['date']}. "
        f"Time: {event['start']} to {event['end']}. Location: {event['location']}. "
        f"Notes: {event['notes']}."
    )


def sync_chromadb():
    events = load_events()
    existing_ids = collection.get().get("ids", [])
    current_ids = {event["id"] for event in events}
    stale_ids = [event_id for event_id in existing_ids if event_id not in current_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)
    if events:
        documents = [event_to_text(event) for event in events]
        embeddings = embedding_model.encode(documents).tolist()
        collection.upsert(
            ids=[event["id"] for event in events],
            documents=documents,
            embeddings=embeddings,
            metadatas=events,
        )


sync_chromadb()


def parse_date(text):
    text = text.lower()
    if "day after tomorrow" in text:
        return BASE_DATE + timedelta(days=2)
    if "tomorrow" in text:
        return BASE_DATE + timedelta(days=1)
    if "today" in text:
        return BASE_DATE

    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for name, weekday_number in weekdays.items():
        if name in text:
            days_ahead = (weekday_number - BASE_DATE.weekday()) % 7
            if f"next {name}" in text and days_ahead == 0:
                days_ahead = 7
            return BASE_DATE + timedelta(days=days_ahead)

    month_match = re.search(r"\b(august|september)\s+(\d{1,2})\b", text)
    if month_match:
        month = 8 if month_match.group(1) == "august" else 9
        return date(2026, month, int(month_match.group(2)))

    iso_match = re.search(r"\b2026-\d{2}-\d{2}\b", text)
    if iso_match:
        return date.fromisoformat(iso_match.group(0))
    return None


def normalize_time(value):
    if not value:
        return ""
    match = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", str(value).lower())
    if not match:
        return ""
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    period = match.group(3)
    if period == "pm" and hour < 12:
        hour += 12
    if period == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return ""
    return f"{hour:02d}:{minute:02d}"


@tool
def get_schedule(query: str, target_date: str = "", start_time: str = "", end_time: str = "") -> str:
    """Retrieve schedule information using ChromaDB semantic search and optional date/time filters. Use for schedule questions, availability, conflicts, or finding an event."""
    events = load_events()
    detected_date = parse_date(query)
    date_filter = target_date or (detected_date.isoformat() if detected_date else "")
    query_text = query or f"schedule {date_filter}"

    retrieved_ids = []
    if collection.count() > 0:
        results = collection.query(
            query_embeddings=embedding_model.encode([query_text]).tolist(),
            n_results=min(10, collection.count()),
        )
        retrieved_ids = results.get("ids", [[]])[0]

    if date_filter:
        candidates = [e for e in events if e["date"] == date_filter]
    else:
        candidates = [e for e in events if e["id"] in retrieved_ids]

    start = normalize_time(start_time)
    end = normalize_time(end_time)
    if start:
        candidates = [e for e in candidates if e["end"] > start]
    if end:
        candidates = [e for e in candidates if e["start"] < end]

    return json.dumps(
        {
            "query": query,
            "date_filter": date_filter,
            "retrieved_by_vector_search": retrieved_ids,
            "results": candidates,
        },
        indent=2,
    )


@tool
def update_schedule(action: str, event_id: str = "", title: str = "", event_type: str = "meeting", event_date: str = "", start_time: str = "", end_time: str = "", location: str = "", notes: str = "") -> str:
    """Add, update, or remove a schedule entry. action must be add, update, or remove."""
    events = load_events()
    action = action.lower().strip()

    if action == "add":
        if not title or not event_date or not start_time:
            return "ERROR: title, event_date and start_time are required."
        start = normalize_time(start_time)
        if not start:
            return "ERROR: invalid start time."
        end = normalize_time(end_time)
        if not end:
            end = f"{(int(start[:2]) + 1) % 24:02d}:{start[3:]}"
        next_number = max([int(e["id"][3:]) for e in events] + [0]) + 1
        new_event = {
            "id": f"evt{next_number:03d}", "title": title, "type": event_type,
            "date": event_date, "start": start, "end": end,
            "location": location, "notes": notes,
        }
        events.append(new_event)
        save_events(events)
        sync_chromadb()
        return json.dumps({"status": "added", "event": new_event}, indent=2)

    if action == "update":
        if not event_id:
            return "ERROR: event_id is required for update."
        for event in events:
            if event["id"] == event_id:
                if title:
                    event["title"] = title
                if event_date:
                    event["date"] = event_date
                if start_time:
                    value = normalize_time(start_time)
                    if not value:
                        return "ERROR: invalid start time."
                    event["start"] = value
                if end_time:
                    value = normalize_time(end_time)
                    if not value:
                        return "ERROR: invalid end time."
                    event["end"] = value
                if event_type:
                    event["type"] = event_type
                if location:
                    event["location"] = location
                if notes:
                    event["notes"] = notes
                save_events(events)
                sync_chromadb()
                return json.dumps({"status": "updated", "event": event}, indent=2)
        return "ERROR: event not found."

    if action == "remove":
        if not event_id:
            return "ERROR: event_id is required for removal."
        remaining = [e for e in events if e["id"] != event_id]
        if len(remaining) == len(events):
            return "ERROR: event not found."
        save_events(remaining)
        sync_chromadb()
        return json.dumps({"status": "removed", "event_id": event_id}, indent=2)

    return "ERROR: action must be add, update, or remove."


TOOLS = [get_schedule, update_schedule]


def get_api_key():
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets["GOOGLE_API_KEY"]).strip()
        except Exception:
            key = ""
    return key


@st.cache_resource
def create_llm(api_key):
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )


def run_agent(user_query):
    api_key = get_api_key()
    if not api_key:
        return "⚠️ Gemini API key is not configured. Add GOOGLE_API_KEY under Streamlit Settings → Secrets."

    llm = create_llm(api_key)
    if llm is None:
        return "⚠️ Could not initialize Gemini."

    system_message = SystemMessage(content=f"""
You are an Agentic RAG Schedule Assistant.
Today's date is {BASE_DATE.isoformat()}.

You have exactly two tools:
1. get_schedule — retrieves existing schedule information from ChromaDB.
2. update_schedule — adds, updates, or removes schedule entries.

Rules:
- Never invent schedule events.
- For schedule questions, call get_schedule.
- For availability questions, call get_schedule first and then determine whether the requested period is free.
- For add requests, call update_schedule with action=add.
- For delete requests, identify the correct event and call update_schedule with action=remove.
- For moving/changing an event, call get_schedule first to identify the event_id, then update_schedule with action=update.
- Convert natural-language dates to YYYY-MM-DD and times to HH:MM for tool arguments.
- After the tool result, answer concisely and clearly.
""")

    messages = [system_message, HumanMessage(content=user_query)]
    model = llm.bind_tools(TOOLS)

    for _ in range(6):
        ai_message = model.invoke(messages)
        messages.append(ai_message)
        tool_calls = getattr(ai_message, "tool_calls", [])

        if not tool_calls:
            text = getattr(ai_message, "text", None)
            if text:
                return text
            if isinstance(ai_message.content, str):
                return ai_message.content
            return str(ai_message.content)

        for tool_call in tool_calls:
            selected_tool = next((t for t in TOOLS if t.name == tool_call["name"]), None)
            if selected_tool is None:
                return f"⚠️ Unknown tool requested: {tool_call['name']}"
            # Pass the complete tool call object so LangChain creates the correct ToolMessage.
            messages.append(selected_tool.invoke(tool_call))

    return "I could not complete the request within the tool-call limit."


st.title("📅 Agentic RAG Schedule Assistant")
st.caption("Gemini + LangChain tool calling + ChromaDB Vector Database")

with st.sidebar:
    st.subheader("System")
    st.write("**LLM:** Gemini 2.5 Flash")
    st.write("**Agent:** LangChain")
    st.write("**Vector DB:** ChromaDB")
    st.write("**Embeddings:** all-MiniLM-L6-v2")
    st.write("**Tools:** get_schedule, update_schedule")
    st.divider()
    st.write(f"**Events:** {len(load_events())}")
    if get_api_key():
        st.success("Gemini API key detected.")
    else:
        st.warning("GOOGLE_API_KEY is not configured.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask about your schedule or make a change...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Agent is checking your schedule..."):
            try:
                answer = run_agent(query)
            except Exception as error:
                answer = (
                    "⚠️ Gemini/Agent error.\n\n"
                    f"`{type(error).__name__}: {error}`\n\n"
                    "Check the Streamlit logs for the full error."
                )
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
