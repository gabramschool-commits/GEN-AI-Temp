import io, hashlib
from typing import List
from pypdf import PdfReader
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# --------------------------
# PDF READER
# --------------------------

def read_pdf_bytes_to_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF using pypdf."""
    text = ""
    try:
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        text = f"[Error reading PDF: {e}]"
    return text

# --------------------------
# TEXT UTILS
# --------------------------
def compute_texts_hash(texts: List[str]) -> str:
    data = "\n".join(texts)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))


# --------------------------
# EMBEDDINGS + FAISS INDEX
# --------------------------
@st.cache_resource(show_spinner=True)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": False},
    )

def build_faiss_index(texts: List[str], chunk_size: int = 800, chunk_overlap: int = 120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents(texts)
    emb = get_embeddings()
    vs = FAISS.from_documents(docs, embedding=emb)
    return vs


# --------------------------
# LLM LOADING
# --------------------------
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_NEW_TOKENS = 256

@st.cache_resource(show_spinner=True)
def load_llm(model_id: str = DEFAULT_MODEL_ID,
             temperature: float = DEFAULT_TEMPERATURE,
             max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        st.warning(f"Failed to load {model_id}, falling back to {FALLBACK_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU only
        do_sample=(temperature > 0.0),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen)


# --------------------------
# RAG CHAIN BUILDER
# --------------------------
SYSTEM_PROMPT = (
    "You are a careful assistant for question answering. Use ONLY the provided context to answer. "
    "If the answer is not in the context, say you don't know. Be concise and cite chunk indices if helpful."
)

def make_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
