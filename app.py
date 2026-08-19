import os,re,json
from datetime import date,timedelta
from pathlib import Path
from typing import TypedDict
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

st.set_page_config(page_title="Agentic RAG Schedule Assistant",page_icon="📅")
DATA=Path(__file__).parent/"schedule.json"
BASE_DATE=date(2026,8,14)
events=json.loads(DATA.read_text(encoding="utf-8"))

@st.cache_resource
def resources():
    client=chromadb.PersistentClient(path=str(Path(__file__).parent/"chroma_db"))
    collection=client.get_or_create_collection("schedule")
    model=SentenceTransformer("all-MiniLM-L6-v2")
    return collection,model

collection,model=resources()

def event_text(e):
    return f"{e['title']} ({e['type']}) on {e['date']} from {e['start']} to {e['end']}, {e['location']}. {e['notes']}"

def sync_index():
    ids=collection.get().get("ids",[])
    if ids: collection.delete(ids=ids)
    if events:
        texts=[event_text(e) for e in events]
        collection.add(ids=[e["id"] for e in events],documents=texts,
                       metadatas=events,embeddings=model.encode(texts).tolist())

sync_index()

def parse_date(text):
    t=text.lower()
    if "today" in t:return BASE_DATE
    if "tomorrow" in t:return BASE_DATE+timedelta(days=1)
    weekdays={"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
    for name,idx in weekdays.items():
        if name in t:return BASE_DATE+timedelta(days=(idx-BASE_DATE.weekday())%7)
    m=re.search(r'(august|september)\s+(\d{1,2})',t)
    if m:return date(2026,8 if m.group(1)=="august" else 9,int(m.group(2)))
    m=re.search(r'(\d{4}-\d{2}-\d{2})',t)
    return date.fromisoformat(m.group(1)) if m else None

def parse_time(text):
    m=re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',text.lower())
    if not m:return None
    h,mi,ap=int(m.group(1)),int(m.group(2) or 0),m.group(3)
    if ap=="pm" and h<12:h+=12
    if ap=="am" and h==12:h=0
    return h*60+mi

def get_schedule(query):
    d=parse_date(query)
    result=collection.query(query_embeddings=model.encode([query]).tolist(),n_results=min(8,max(1,collection.count())))
    ids=result.get("ids",[[]])[0]
    found=[e for e in events if e["id"] in ids]
    if d:
        found=[e for e in found if e["date"]==d.isoformat()]
        if not found: found=[e for e in events if e["date"]==d.isoformat()]
    return sorted(found,key=lambda x:(x["date"],x["start"]))

def update_schedule(action,**kwargs):
    global events
    if action=="add":
        new={"id":"evt"+str(max([int(e["id"][3:]) for e in events]+[0])+1).zfill(3),
             "title":kwargs["title"],"type":kwargs.get("type","meeting"),
             "date":kwargs["date"],"start":kwargs["start"],"end":kwargs.get("end",kwargs["start"]),
             "location":kwargs.get("location",""),"notes":kwargs.get("notes","")}
        events.append(new)
    elif action=="remove":
        events=[e for e in events if e["id"]!=kwargs["id"]]
        new={"removed":kwargs["id"]}
        DATA.write_text(json.dumps(events,indent=2),encoding="utf-8"); sync_index(); return new
    elif action=="update":
        new=None
        for e in events:
            if e["id"]==kwargs["id"]:
                for k,v in kwargs.items():
                    if k!="id" and v is not None:e[k]=v
                new=e
                break
        if new is None:return {"error":"Event not found"}
    DATA.write_text(json.dumps(events,indent=2),encoding="utf-8"); sync_index()
    return new

class State(TypedDict,total=False):
    query:str
    action:str
    result:object
    answer:str

def route(state):
    q=state["query"].lower()
    if any(x in q for x in ["add ","schedule ","create ","book "]): return {"action":"add"}
    if any(x in q for x in ["move ","change ","reschedule "]): return {"action":"update"}
    if any(x in q for x in ["remove","delete","cancel"]): return {"action":"remove"}
    return {"action":"get"}

def tool(state):
    q=state["query"]; action=state["action"]; d=parse_date(q)
    if action=="get":
        return {"result":get_schedule(q)}
    if action=="add":
        tm=re.search(r'at\s+(.+)',q.lower())
        mins=parse_time(tm.group(1)) if tm else 9*60
        start=f"{mins//60:02d}:{mins%60:02d}"
        title=re.sub(r'^(add|schedule|create|book)\s+(a\s+|an\s+)?','',q,flags=re.I)
        title=re.split(r'\s+(?:on|for)\s+',title,flags=re.I)[0].strip().title() or "New Meeting"
        return {"result":update_schedule("add",title=title,date=(d or BASE_DATE).isoformat(),
                                         start=start,end=f"{(mins//60+1)%24:02d}:{mins%60:02d}")}
    found=get_schedule(q)
    if not found:return {"result":[]}
    if action=="remove":return {"result":update_schedule("remove",id=found[0]["id"])}
    new_matches=re.findall(r'(?:to|at)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)',q.lower())
    new=parse_time(new_matches[-1]) if new_matches else None
    if new is None:return {"result":found}
    return {"result":update_schedule("update",id=found[0]["id"],start=f"{new//60:02d}:{new%60:02d}")}

def respond(state):
    result=state.get("result")
    key=os.getenv("GROQ_API_KEY")
    if not key:
        if isinstance(result,list):
            text="No matching schedule entries found." if not result else "\n".join(
                f"• {e['date']} {e['start']}-{e['end']} — {e['title']}" for e in result)
        else:text=f"Done: {result}"
        return {"answer":text+"\n\nAdd GROQ_API_KEY to enable the LLM response."}
    llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0.1,api_key=key)
    prompt=f"""You are a simple schedule assistant. Answer the user's request using only the tool result.
User: {state['query']}
Tool result: {json.dumps(result,default=str)}
Be concise. If an event was added, changed, or removed, clearly confirm it. Never invent events."""
    return {"answer":llm.invoke(prompt).content}

def build_graph():
    g=StateGraph(State)
    g.add_node("route",route);g.add_node("tool",tool);g.add_node("respond",respond)
    g.add_edge(START,"route");g.add_edge("route","tool");g.add_edge("tool","respond");g.add_edge("respond",END)
    return g.compile()

graph=build_graph()
st.title("📅 Agentic RAG Schedule Assistant")
st.caption("LangGraph + LangChain LLM + ChromaDB RAG + simple schedule tools")

if "messages" not in st.session_state:st.session_state.messages=[]
for m in st.session_state.messages:st.chat_message(m["role"]).write(m["content"])
q=st.chat_input("Ask about your schedule or make a change...")
if q:
    st.session_state.messages.append({"role":"user","content":q})
    with st.spinner("Agent is routing → using a tool → generating an answer..."):
        a=graph.invoke({"query":q})
    st.session_state.messages.append({"role":"assistant","content":a["answer"]})
    st.rerun()

with st.sidebar:
    st.subheader("Agent tools")
    st.code("get_schedule()\nupdate_schedule(add/update/remove)")
    st.write(f"Events stored: {len(events)}")
