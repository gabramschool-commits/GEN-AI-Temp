---
title: SimpleQnA
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---
# Simple QA with LangChain (Open-Source LLM)
This Space demonstrates a minimal Retrieval-Augmented Generation (RAG) app:
- Upload PDFs or paste text
- Build a FAISS index with MiniLM embeddings
- Ask grounded questions with an open-source LLM (default: `Qwen/Qwen2.5-1.5B-Instruct`)
## How to use
1. Upload a PDF or paste text in **Knowledge Base**.
2. Click **Build Knowledge Base**.
3. Type a question and click **Get Answer**.
4. (Optional) Toggle **Show retrieved chunks** to inspect sources.
## Swap the model
Change the model in the sidebar (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
