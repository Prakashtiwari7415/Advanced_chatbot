"""Streamlit interface for the Advanced Chatbot."""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chatbot_backend import (
    chatbot,
    clear_knowledge_base,
    ingest_document,
    knowledge_base_status,
    retrieve_all_threads,
)


st.set_page_config(
    page_title="Kay.. · Advanced Chatbot",
    page_icon=":material/auto_awesome:",
    layout="centered",
    initial_sidebar_state="expanded",
)


def content_as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item if isinstance(item, str) else item.get("text", "")
            for item in value
            if isinstance(item, str)
            or (isinstance(item, dict) and item.get("type") == "text")
        )
    return str(value or "")


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state.threads:
        st.session_state.threads.append(thread_id)


def load_thread(thread_id: str) -> list[dict[str, str]]:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = []
    for message in state.values.get("messages", []):
        if isinstance(message, (HumanMessage, AIMessage)) and (body := content_as_text(message.content)):
            messages.append(
                {"role": "user" if isinstance(message, HumanMessage) else "assistant", "content": body}
            )
    return messages


def thread_label(thread_id: str) -> str:
    for message in load_thread(thread_id):
        if message["role"] == "user":
            preview = message["content"].replace("\n", " ")
            return preview[:32] + ("…" if len(preview) > 32 else "")
    return "New conversation"


def start_new_chat() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    add_thread(st.session_state.thread_id)


@st.dialog("Clear knowledge base")
def confirm_clear_knowledge_base() -> None:
    st.warning("This permanently removes every indexed document chunk.")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Cancel"):
            st.rerun()
        if st.button("Clear all", icon=":material/delete_forever:", type="primary"):
            clear_knowledge_base()
            st.toast("Knowledge base cleared", icon=":material/delete_sweep:")
            st.rerun()


def response_stream(prompt: str):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    for message, _ in chatbot.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(message, ToolMessage):
            name = getattr(message, "name", "tool").replace("_", " ")
            with st.status(f"Used {name}", state="complete", expanded=False):
                st.caption("Preparing a grounded answer.")
        elif isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            if body := content_as_text(message.content):
                yield body


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "threads" not in st.session_state:
    st.session_state.threads = retrieve_all_threads()
if "messages" not in st.session_state:
    st.session_state.messages = []
add_thread(st.session_state.thread_id)


with st.sidebar:
    st.title("Kay...", icon=":material/auto_awesome:", anchor=False)
    st.caption("Your research-ready AI assistant")
    st.badge("Online", icon=":material/check_circle:", color="green")
    st.space("small")
    st.button(
        "New chat",
        icon=":material/edit_square:",
        type="primary",
        width="stretch",
        on_click=start_new_chat,
    )

    st.subheader("Knowledge base", icon=":material/database:", anchor=False)
    summary = knowledge_base_status()
    with st.container(border=True):
        st.markdown(
            f":violet-badge[{summary['sources']} sources] "
            f":blue-badge[{summary['chunks']} chunks]"
        )
        st.caption("Upload references for citation-backed answers.")
        uploads = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploads and st.button(
            "Add to knowledge base",
            icon=":material/database_upload:",
            width="stretch",
        ):
            indexed, notices = 0, []
            with st.spinner("Indexing documents…"):
                for upload in uploads:
                    try:
                        count = ingest_document(upload.name, upload.getvalue())
                        indexed += count
                        if not count:
                            notices.append(f"{upload.name} is already indexed.")
                    except Exception as exc:
                        notices.append(f"{upload.name}: {exc}")
            if indexed:
                st.toast(f"Added {indexed} knowledge chunks", icon=":material/check_circle:")
            for notice in notices:
                st.warning(notice)
            st.rerun()
        if summary["chunks"]:
            if st.button("Clear all documents", icon=":material/delete_sweep:", width="stretch"):
                confirm_clear_knowledge_base()

    st.subheader("Recent chats", icon=":material/history:", anchor=False)
    for thread_id in reversed(st.session_state.threads):
        if st.button(
            thread_label(thread_id),
            key=f"thread_{thread_id}",
            icon=":material/chat_bubble:" if thread_id != st.session_state.thread_id else ":material/forum:",
            type="primary" if thread_id == st.session_state.thread_id else "secondary",
            width="stretch",
        ):
            st.session_state.thread_id = thread_id
            st.session_state.messages = load_thread(thread_id)
            st.rerun()


with st.container(horizontal=True, horizontal_alignment="distribute"):
    st.subheader("Kay..", icon=":material/auto_awesome:", anchor=False)
    st.badge("RAG ready", icon=":material/verified:", color="violet")

if st.session_state.messages:
    st.caption("Answers can use your knowledge base, live web search, and specialist tools.")
else:
    st.write("### Hi, I’m Kay..")
    st.caption("Ask a question, search the web, or upload documents to create your private knowledge base.")
    selected = st.pills(
        "Start with a prompt",
        [
            "What can you help me with?",
            "What time is it in Asia/Kolkata?",
            "How do I use document search?",
        ],
        label_visibility="collapsed",
    )

for message in st.session_state.messages:
    avatar = ":material/person:" if message["role"] == "user" else ":material/auto_awesome:"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

chat_prompt = st.chat_input("Message Kay..", submit_mode="disable")
prompt = selected if not st.session_state.messages and selected else chat_prompt
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        try:
            answer = st.write_stream(response_stream(prompt))
        except Exception as exc:
            st.error("I couldn't complete that request. Check your API keys and try again.")
            st.exception(exc)
            answer = ""
    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})
