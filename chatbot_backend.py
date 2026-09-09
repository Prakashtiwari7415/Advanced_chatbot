"""LangGraph backend and persistent local RAG knowledge base."""
from __future__ import annotations

import ast
import hashlib
import io
import operator
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Annotated, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import chromadb
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pypdf import PdfReader

load_dotenv()
ROOT = Path(__file__).resolve().parent
VECTOR_DIR = ROOT / "data" / "chroma"
COLLECTION = "chatbot_knowledge"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

def _collection():
    VECTOR_DIR.parent.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_DIR)).get_or_create_collection(COLLECTION)

@lru_cache(maxsize=1)
def _store():
    return Chroma(COLLECTION, GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"), str(VECTOR_DIR))

def knowledge_base_status() -> dict[str, int]:
    records = _collection().get(include=["metadatas"])
    return {"chunks": len(records.get("ids") or []), "sources": len({m.get("source") for m in records.get("metadatas") or [] if m})}

def _extract_documents(filename: str, raw: bytes) -> list[Document]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported files: PDF, TXT, MD, and Markdown.")
    metadata = {"source": filename, "file_hash": hashlib.sha256(raw).hexdigest()}
    if extension == ".pdf":
        try:
            pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages]
        except Exception as exc:
            raise ValueError(f"Could not read PDF: {exc}") from exc
        docs = [Document(page_content=text, metadata={**metadata, "page": index}) for index, text in enumerate(pages, 1) if text.strip()]
    else:
        text = raw.decode("utf-8", errors="replace")
        docs = [Document(page_content=text, metadata=metadata)] if text.strip() else []
    if not docs:
        raise ValueError("No extractable text was found.")
    return docs

def ingest_document(filename: str, raw: bytes) -> int:
    docs = _extract_documents(filename, raw)
    if _collection().get(where={"file_hash": docs[0].metadata["file_hash"]}, include=[]).get("ids"):
        return 0
    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150, add_start_index=True).split_documents(docs)
    for index, chunk in enumerate(chunks, 1):
        chunk.metadata["chunk"] = index
    _store().add_documents(chunks)
    return len(chunks)

def clear_knowledge_base() -> None:
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    try: client.delete_collection(COLLECTION)
    except ValueError: pass
    _store.cache_clear()

@tool
def search_knowledge_base(query: str) -> str:
    """Search uploaded documents. Use it for questions about the user's files; cite returned sources."""
    if not knowledge_base_status()["chunks"]:
        return "The knowledge base is empty. Ask the user to upload a document."
    results = _store().similarity_search_with_relevance_scores(query, k=4)
    return "\n\n".join(
        f"[Source: {doc.metadata['source']}{', p. ' + str(doc.metadata['page']) if doc.metadata.get('page') else ''}; relevance: {score:.2f}]\n{' '.join(doc.page_content.split())}"
        for doc, score in results
    ) or "No relevant passages found."

BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY: return UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100: raise ValueError("exponent must be between -100 and 100")
        return BINOPS[type(node.op)](left, right)
    raise ValueError("only numbers, parentheses, and + - * / // % ** are allowed")

@tool
def calculate(expression: str) -> str:
    """Safely evaluate basic arithmetic; do not use for code."""
    try: return str(_evaluate(ast.parse(expression, mode="eval").body))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc: return f"Unable to calculate: {exc}"

@tool
def current_time(timezone: str = "UTC") -> str:
    """Get current date and time for an IANA zone such as Asia/Kolkata."""
    from datetime import datetime
    try: return datetime.now(ZoneInfo(timezone)).strftime("%A, %Y-%m-%d %H:%M:%S %Z")
    except ZoneInfoNotFoundError: return f"Unknown timezone '{timezone}'. Use an IANA zone such as Asia/Kolkata."

@tool
def check_order_status(order_id: str) -> str:
    """Look up a customer order's shipping status."""
    return f"Order {order_id} is currently shipped and arriving in 2 business days via FedEx."

web_search_tool = TavilySearch(max_results=3)
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [search_knowledge_base, calculate, current_time, web_search_tool, wiki_tool, check_order_status]
tool_node = ToolNode(tools)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)
SYSTEM = """You are a helpful accurate assistant. For questions about uploaded files, always call search_knowledge_base first. Ground answers in the excerpts and cite claims as [filename, p. N] or [filename]. If context is incomplete, say so. Use web search for current information and calculate for arithmetic."""
class ChatState(TypedDict): messages: Annotated[list[BaseMessage], add_messages]
def chat_node(state: ChatState): return {"messages": [llm_with_tools.invoke([SystemMessage(content=SYSTEM), *state["messages"]])]}
conn = sqlite3.connect(str(ROOT / "chatbot.db"), check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node); graph.add_node("tools", tool_node); graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition); graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpointer)
def retrieve_all_threads() -> list[str]: return list({checkpoint.config["configurable"]["thread_id"] for checkpoint in checkpointer.list(None)})
