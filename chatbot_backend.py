from langgraph.graph import StateGraph, START, END ,MessagesState
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition

load_dotenv()
# 1. Define standard built-in utility tools
web_search_tool = TavilySearchResults(max_results=3) # Requires TAVILY_API_KEY environment variable
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# 2. Define a custom customer-specific tool (e.g., Checking internal systems)
@tool
def check_order_status(order_id: str) -> str:
    """Use this tool to lookup the shipping and tracking status of a customer order."""
    # In production, you would make an API or DB call here
    return f"Order {order_id} is currently shipped and arriving in 2 business days via FedEx."

# 3. Combine your tools into a list
tools = [web_search_tool, wiki_tool, check_order_status]

# 4. Instantiate the tool execution node for LangGraph
tool_node = ToolNode(tools)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_conditional_edges(
    "chat_node",           # Or whatever your call_model node name is
    tools_condition,   # Routes to "tools" if Gemini requests a tool, otherwise loops to END
)
graph.add_edge("tools","chat_node")
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

