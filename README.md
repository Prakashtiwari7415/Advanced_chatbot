# Advanced Chatbot

Streamlit chatbot using Gemini and LangGraph, with persistent chat history, local RAG, web/Wikipedia search, calculator, timezone lookup, and an order-status example tool.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add `.env`:

```env
GOOGLE_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key
```

`TAVILY_API_KEY` is only required for web search. Run `streamlit run chatbot_frontend.py`.

## RAG

Upload PDF, TXT, or Markdown in the sidebar, then index it. Files are chunked and embedded with Gemini `gemini-embedding-001`; the persistent index is stored in `data/chroma`. Raw uploads are not kept. The chatbot searches the knowledge base for file questions and cites sources as `[filename, p. N]` where possible.
