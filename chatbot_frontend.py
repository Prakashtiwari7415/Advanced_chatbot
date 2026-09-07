import streamlit as st
from chatbot_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

# **************************************** UI Config & Custom Styling *****************
st.set_page_config(
    page_title="LangGraph Conversational AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism & Modern Chat CSS Injection
st.markdown("""
    <style>
    /* Main App Background Styling */
    .stApp {
        background: linear-gradient(135deg, #12131C 0%, #1A1C29 100%);
        color: #E2E8F0;
    }
    
    /* Sidebar styling refinement */
    section[data-testid="stSidebar"] {
        background-color: #0E1017 !important;
        border-right: 1px solid #2D3142;
    }
    
    /* Custom Stylings for User vs Assistant Bubbles */
    .chat-bubble {
        padding: 14px 18px;
        border-radius: 18px;
        margin-bottom: 15px;
        max-width: 80%;
        line-height: 1.5;
        font-size: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        animation: fadeIn 0.3s ease-in-out;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-bubble {
        background: #222533;
        color: #E2E8F0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border: 1px solid #31354A;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Prettify Sidebar Buttons */
    div[data-testid="stMarkdownContainer"] > h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# **************************************** Utility Functions *************************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def get_thread_label(thread_id):
    """Extracts a neat preview title from the conversation history."""
    messages = load_conversation(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            return msg.content[:22] + "..." if len(msg.content) > 22 else msg.content
    return f"✨ Loading...."

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

with st.sidebar:
    st.markdown("### 🤖 LangGraph Agent")
    st.write("---")
    
    # Large accent color action button
    if st.button('➕ Start New Session', use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()

    st.markdown("#### 💬 Active Conversations")
    
    # Render historical conversation elements cleanly
    for thread_id in st.session_state['chat_threads'][::-1]:
        button_label = get_thread_label(thread_id)
        
        # Highlight active conversation using native container context logic
        is_active = (st.session_state['thread_id'] == thread_id)
        btn_type = "secondary" if not is_active else "primary"
        
        if st.button(button_label, key=f"btn_{thread_id}", use_container_width=True, type=btn_type):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []
            for msg in messages:
                role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages
            st.rerun()

# **************************************** Main UI Canvas *****************************

# Application header layout block
st.title("⚡ Dynamic Intelligence Interface")
st.caption(f"Secure Thread Routing ID: `{st.session_state['thread_id']}`")
st.write("---")

# Container rendering logic using custom beautiful HTML templates
for message in st.session_state['message_history']:
    if message['role'] == 'user':
        st.markdown(
            f'<div class="chat-bubble user-bubble">🧑‍💻 <b>You:</b><br>{message["content"]}</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-bubble assistant-bubble">🤖 <b>Agent:</b><br>{message["content"]}</div>', 
            unsafe_allow_html=True
        )

# Interaction interface
user_input = st.chat_input('Send a message to the agent...')

if user_input:
    # 1. Append and render User Prompt immediately
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    st.markdown(
        f'<div class="chat-bubble user-bubble">🧑‍💻 <b>You:</b><br>{user_input}</div>', 
        unsafe_allow_html=True
    )

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # 2. Render Assistant response area dynamically using Streamlit markdown injections
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    st.rerun()
