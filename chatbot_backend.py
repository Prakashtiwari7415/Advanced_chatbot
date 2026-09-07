from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver #INmomery saver is replaced with sqlite saver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  ##add_masseges is a resucer to handle linearity of messages

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


#we are connecting our database with sqlite
connt=sqlite3.connect(databse='chatbot.db',check_same_thread=False) #Here false means we are saying that we will use this database for dufferent different thread



# Checkpointer
checkpointer = SqliteSaver(conn=connt)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

## we are making function to connect our sqlite database with our frontend
def retrieve_all_threads():
    all_threads=set()
    for checkpointer in checkpointer.list(None):
        all_threads.add(checkpointer.config['configurable']['thread_id'])

    return list(all_threads)
     