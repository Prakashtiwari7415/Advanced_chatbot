import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from chatbot_backend import chatbot, clear_knowledge_base, ingest_document, knowledge_base_status, retrieve_all_threads

st.set_page_config(page_title="Advanced Chatbot", page_icon=":material/smart_toy:", layout="wide")

def text(value):
    if isinstance(value, str): return value
    if isinstance(value, list): return "".join(i if isinstance(i, str) else i.get("text", "") for i in value if isinstance(i, str) or (isinstance(i, dict) and i.get("type") == "text"))
    return str(value or "")
def add_thread(thread_id):
    if thread_id not in st.session_state.threads: st.session_state.threads.append(thread_id)
def load_thread(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": text(m.content)} for m in state.values.get("messages", []) if isinstance(m, (HumanMessage, AIMessage)) and text(m.content)]
def label(thread_id):
    for item in load_thread(thread_id):
        if item["role"] == "user": return item["content"][:30] + ("…" if len(item["content"]) > 30 else "")
    return "New conversation"
def response_stream(prompt):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    for message, _ in chatbot.stream({"messages": [HumanMessage(content=prompt)]}, config=config, stream_mode="messages"):
        if isinstance(message, ToolMessage):
            with st.status(f"Used {getattr(message, 'name', 'tool').replace('_', ' ')}", state="complete", expanded=False): st.caption("The agent used a tool to prepare this answer.")
        elif isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            content = text(message.content)
            if content: yield content

if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "threads" not in st.session_state: st.session_state.threads = retrieve_all_threads()
if "messages" not in st.session_state: st.session_state.messages = []
add_thread(st.session_state.thread_id)

with st.sidebar:
    st.title("Advanced Chatbot", anchor=False)
    st.caption("Gemini + LangGraph + local RAG")
    if st.button("New conversation", icon=":material/add:", type="primary", width="stretch"):
        st.session_state.thread_id, st.session_state.messages = str(uuid.uuid4()), []
        add_thread(st.session_state.thread_id); st.rerun()
    st.divider(); st.subheader("Knowledge base", anchor=False)
    info = knowledge_base_status(); st.caption(f"{info['sources']} document(s) · {info['chunks']} indexed chunks")
    files = st.file_uploader("Add reference documents", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True, help="Files are indexed locally; raw uploads are not kept.")
    if files and st.button("Index documents", icon=":material/database_upload:", width="stretch"):
        indexed, notices = 0, []
        with st.spinner("Reading and indexing…"):
            for file in files:
                try:
                    count = ingest_document(file.name, file.getvalue()); indexed += count
                    if not count: notices.append(f"{file.name} was already indexed.")
                except Exception as exc: notices.append(f"{file.name}: {exc}")
        if indexed: st.success(f"Indexed {indexed} chunks.")
        for notice in notices: st.warning(notice)
        st.rerun()
    if info["chunks"] and st.button("Clear knowledge base", icon=":material/delete_sweep:", width="stretch"):
        clear_knowledge_base(); st.rerun()
    st.divider(); st.subheader("Conversations", anchor=False)
    for thread_id in reversed(st.session_state.threads):
        if st.button(label(thread_id), key=f"thread_{thread_id}", type="primary" if thread_id == st.session_state.thread_id else "secondary", width="stretch"):
            st.session_state.thread_id, st.session_state.messages = thread_id, load_thread(thread_id); st.rerun()

st.title("Ask, search, and reason", anchor=False)
st.caption("Upload PDF, TXT, or Markdown documents to receive grounded answers with citations.")
if not st.session_state.messages: st.info("Try: “Summarize the uploaded policy” or “What time is it in Asia/Kolkata?”")
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])
if prompt := st.chat_input("Ask a question", submit_mode="disable"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        try: answer = st.write_stream(response_stream(prompt))
        except Exception as exc:
            st.error("I couldn't complete that request. Check your API keys and try again."); st.exception(exc); answer = ""
    if answer: st.session_state.messages.append({"role": "assistant", "content": answer})
