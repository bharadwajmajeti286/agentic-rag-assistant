
import json, re, uuid
from datetime import datetime, date, timedelta
from pathlib import Path
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
SCHEDULE_FILE = BASE / "schedule.json"
DB_DIR = BASE / "chroma_db"

st.set_page_config(page_title="Agentic RAG Schedule Assistant", page_icon="📅", layout="wide")

@st.cache_resource
def load_components():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name="schedule",
        metadata={"hnsw:space": "cosine"}
    )
    return model, collection

model, collection = load_components()

def load_schedule():
    return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))

def save_schedule(events):
    SCHEDULE_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")

def event_text(e):
    return f"{e['date']} {e['start']}-{e['end']} | {e['title']} | {e['type']} | {e['description']}"

def rebuild_vector_store():
    events = load_schedule()
    ids = [e["id"] for e in events]
    if ids:
        try:
            collection.delete(ids=ids)
        except Exception:
            pass
        collection.add(
            ids=ids,
            documents=[event_text(e) for e in events],
            metadatas=[
                {"date": e["date"], "start": e["start"], "end": e["end"],
                 "title": e["title"], "type": e["type"]}
                for e in events
            ]
        )

@st.cache_resource
def initialize_store():
    rebuild_vector_store()
    return True

initialize_store()

def get_schedule(query, target_date=None, start_time=None, end_time=None, n_results=8):
    """Tool 1: retrieve relevant schedule information."""
    q = query or "schedule"
    where = None
    if target_date:
        where = {"date": target_date}
    try:
        result = collection.query(
            query_embeddings=[model.encode(q).tolist()],
            n_results=n_results,
            where=where
        )
        ids = result.get("ids", [[]])[0]
        events = load_schedule()
        by_id = {e["id"]: e for e in events}
        found = [by_id[i] for i in ids if i in by_id]
    except Exception:
        found = []

    # Exact time filtering is applied after semantic retrieval.
    if target_date:
        found = [e for e in found if e["date"] == target_date]
    if start_time and end_time:
        found = [e for e in found if not (e["end"] <= start_time or e["start"] >= end_time)]
    return found

def update_schedule(action, event=None, event_id=None):
    """Tool 2: add, update, or remove schedule entries."""
    events = load_schedule()
    if action == "add":
        event = dict(event)
        event["id"] = event.get("id") or f"evt_{uuid.uuid4().hex[:8]}"
        events.append(event)
    elif action == "update":
        for i, e in enumerate(events):
            if e["id"] == event_id:
                events[i] = {**e, **event}
                break
        else:
            return {"ok": False, "message": "Event not found."}
    elif action == "remove":
        old = len(events)
        events = [e for e in events if e["id"] != event_id]
        if len(events) == old:
            return {"ok": False, "message": "Event not found."}
    else:
        return {"ok": False, "message": "Unsupported action."}
    events.sort(key=lambda x: (x["date"], x["start"]))
    save_schedule(events)
    rebuild_vector_store()
    return {"ok": True, "message": f"Schedule {action}d successfully."}

def parse_date(text):
    t = text.lower()
    today = date.today()
    if "today" in t: return today.isoformat()
    if "tomorrow" in t: return (today + timedelta(days=1)).isoformat()
    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for i, wd in enumerate(weekdays):
        if wd in t:
            delta = (i - today.weekday()) % 7
            if delta == 0: delta = 7
            return (today + timedelta(days=delta)).isoformat()
    m = re.search(r"\b(august|september)\s+(\d{1,2})\b", t)
    if m:
        month = 8 if m.group(1) == "august" else 9
        return date(2026, month, int(m.group(2))).isoformat()
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
    return m.group(1) if m else None

def parse_time_range(text):
    t = text.lower()
    if "afternoon" in t: return ("12:00","17:00")
    if "morning" in t: return ("08:00","12:00")
    if "evening" in t: return ("17:00","22:00")
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
    if not m: return None, None
    h = int(m.group(1)); minute = int(m.group(2) or 0); ap = m.group(3)
    if ap == "pm" and h < 12: h += 12
    if ap == "am" and h == 12: h = 0
    start = f"{h:02d}:{minute:02d}"
    return start, f"{h+1:02d}:{minute:02d}" if h < 23 else "23:59"

def find_matching_event(text):
    events = get_schedule(text, parse_date(text))
    low = text.lower()
    title_words = [w for w in re.findall(r"[a-z]+", low) if len(w) > 3]
    scored = []
    for e in events:
        score = sum(w in e["title"].lower() or w in e["description"].lower() for w in title_words)
        scored.append((score, e))
    return max(scored, key=lambda x: x[0])[1] if scored else None

def agent(user_query):
    """Simple agent/router: decides whether to retrieve or modify schedule."""
    q = user_query.lower().strip()
    if any(x in q for x in ["add ", "schedule ", "create ", "book ", "set a meeting"]):
        # Basic add parser
        d = parse_date(q) or date.today().isoformat()
        tm, _ = parse_time_range(q)
        title = "New Meeting"
        m = re.search(r"(?:add|schedule|create|book)\s+(?:a\s+)?(.+?)(?:\s+on\s+|\s+at\s+|\s*$)", q)
        if m:
            title = m.group(1).strip().title()
        if not tm:
            tm = "15:00"
        h, mi = map(int, tm.split(":"))
        end = f"{h+1:02d}:{mi:02d}" if h < 23 else "23:59"
        result = update_schedule("add", {
            "date": d, "start": tm, "end": end, "title": title,
            "type": "meeting", "description": "Added through the schedule assistant."
        })
        return result["message"]

    if any(x in q for x in ["remove ", "delete ", "cancel "]):
        e = find_matching_event(q)
        if not e: return "I couldn't identify the event to remove."
        result = update_schedule("remove", event_id=e["id"])
        return result["message"]

    if any(x in q for x in ["move ", "reschedule ", "change "]):
        e = find_matching_event(q)
        if not e: return "I couldn't identify the event to update."
        times = re.findall(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", q)
        if len(times) >= 1:
            raw = times[-1]
            h = int(raw[0]); mi = int(raw[1] or 0); ap = raw[2]
            if ap == "pm" and h < 12: h += 12
            if ap == "am" and h == 12: h = 0
            new_start = f"{h:02d}:{mi:02d}"
            dur = 60
            old_h, old_m = map(int, e["start"].split(":"))
            old_end_h, old_end_m = map(int, e["end"].split(":"))
            dur = (old_end_h*60+old_end_m)-(old_h*60+old_m)
            total = h*60+mi+dur
            new_end = f"{total//60:02d}:{total%60:02d}"
            result = update_schedule("update", {"start":new_start,"end":new_end}, e["id"])
            return result["message"]
        return "Please specify the new time."

    d = parse_date(q)
    start, end = parse_time_range(q)
    results = get_schedule(q, d, start, end)
    if not results:
        return "You're free during that requested period based on the current schedule."
    results.sort(key=lambda e: (e["date"], e["start"]))
    lines = ["Here is what I found:"]
    for e in results:
        lines.append(f"• {e['date']} {e['start']}-{e['end']} — {e['title']} ({e['type']})")
    return "\n".join(lines)

st.title("📅 Agentic RAG Schedule Assistant")
st.caption("30-day schedule • ChromaDB vector retrieval • get_schedule + update_schedule tools")

with st.sidebar:
    st.header("Examples")
    st.write("• What do I have scheduled tomorrow?")
    st.write("• Am I free Friday afternoon?")
    st.write("• Add a meeting on August 15 at 3 PM.")
    st.write("• Move my meeting from 2 PM to 4 PM.")
    st.divider()
    st.write("Vector DB: ChromaDB")
    st.write("Embeddings: all-MiniLM-L6-v2")

query = st.chat_input("Ask about your schedule or make a change...")
if query:
    st.chat_message("user").write(query)
    response = agent(query)
    st.chat_message("assistant").write(response)

st.subheader("Current 30-day schedule")
events = load_schedule()
st.dataframe(events, use_container_width=True, hide_index=True)
