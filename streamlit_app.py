import os, pathlib, io, hashlib
from typing import List
import streamlit as st

# Core LLM + embeddings + vector DB
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain core graph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter
# PDF parsing
from pypdf import PdfReader


# --------------------------
# Cache setup (safe fallback)
# --------------------------
def _ensure_dir(p: str) -> str:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    return p

HF_CACHE_DIR = None
for candidate in (
    os.getenv("HF_CACHE_DIR"),
    "/data/.cache/huggingface",  # persistent on Spaces
    os.path.expanduser("~/.cache/huggingface"),
    "/home/user/.cache/huggingface",
    "/tmp/hf_cache",
):
    if not candidate:
        continue
    try:
        HF_CACHE_DIR = _ensure_dir(candidate)
        break
    except Exception:
        continue

if HF_CACHE_DIR is None:
    HF_CACHE_DIR = _ensure_dir("/tmp/hf_cache")

# Set all relevant caches
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = _ensure_dir(os.path.join(HF_CACHE_DIR, "transformers"))
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _ensure_dir(os.path.join(HF_CACHE_DIR, "sentence_transformers"))
os.environ["TORCH_HOME"] = _ensure_dir(os.path.join(HF_CACHE_DIR, "torch"))


# --------------------------
# App Config
# --------------------------
st.set_page_config(page_title="Simple QA with LangChain (Open LLM)", page_icon="🤖", layout="wide")

# 🔄 Model swapping guidance
# - Swap models by editing DEFAULT_MODEL_ID or entering a Hugging Face repo in the sidebar
# - Good CPU options:
#   • Qwen/Qwen2.5-1.5B-Instruct (default, strong)
#   • TinyLlama/TinyLlama-1.1B-Chat-v1.0 (very light)
# - Experimental: microsoft/bitnet-b1.58-2B-4T (efficient, but may require patched Transformers)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # default, strong
FALLBACK_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
SYSTEM_PROMPT = (
    "You are a careful assistant for question answering. Use ONLY the provided context to answer. "
    "If the answer is not in the context, say you don't know. Be concise and cite chunk indices if helpful."
)


# --------------------------
# Utilities
# --------------------------
def read_pdf_bytes_to_text(file_like: io.BytesIO) -> str:
    file_like.seek(0)
    reader = PdfReader(file_like)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def compute_texts_hash(texts: List[str]) -> str:
    data = "\n".join(texts)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))


# --------------------------
# Caches
# --------------------------
@st.cache_resource(show_spinner=True)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
        model_kwargs={"local_files_only": False},
    )


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
    except Exception:
        st.warning(f"⚠️ Failed to load {model_id}, falling back to {FALLBACK_MODEL_ID}")
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
        device=-1,  # force CPU (safe on Spaces). To use GPU: device=0 if torch.cuda.is_available() else -1
        do_sample=(temperature > 0.0),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen)


def build_faiss_index(texts: List[str], chunk_size: int = 800, chunk_overlap: int = 120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents(texts)
    emb = get_embeddings()
    vs = FAISS.from_documents(docs, embedding=emb)
    return vs


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


# --------------------------
# UI
# --------------------------
st.title("Simple QA with LangChain (Open-Source LLM)")

with st.sidebar:
    st.header("Model & Generation")
    model_id = st.text_input(
        "Model ID",
        value=DEFAULT_MODEL_ID,
        help=(
            "Swap models by entering a Hugging Face repo.\n"
            "Good CPU picks:\n"
            "- Qwen/Qwen2.5-1.5B-Instruct (default, strong)\n"
            "- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (very light)\n\n"
            "⚠️ BitNet (microsoft/bitnet-b1.58-2B-4T) is efficient but may require custom setup."
        )
    )
    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05)
    max_new_tokens = st.slider("Max new tokens", 32, 1024, DEFAULT_MAX_NEW_TOKENS, 32)

    st.header("Chunking")
    chunk_size = st.slider("Chunk size (chars)", 200, 1500, 800, 50,
                           help="Keep chunks near 800 chars with overlap ~120 for speed/quality balance.")
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 120, 10)

    if st.button("Reload LLM"):
        st.session_state.pop("llm", None)


# Inputs
left, right = st.columns([1, 1])

with left:
    st.subheader("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDFs or .txt files", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    pasted_text = st.text_area("Or paste reference text here", height=180, placeholder="Paste context...")

    if st.button("Build Knowledge Base"):
        texts: List[str] = []
        if pasted_text and pasted_text.strip():
            texts.append(pasted_text.strip())
        if uploaded_files:
            for f in uploaded_files:
                if f.name.lower().endswith(".pdf"):
                    texts.append(read_pdf_bytes_to_text(f))
                else:
                    content = f.read().decode("utf-8", errors="ignore")
                    texts.append(content)

        if not texts:
            st.warning("Please paste text or upload at least one file.")
        else:
            kb_hash = compute_texts_hash(texts)
            with st.spinner("Embedding & indexing (FAISS)…"):
                vs = build_faiss_index(texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.session_state["kb_hash"] = kb_hash
            st.session_state["vectorstore"] = vs
            st.success(f"Knowledge base ready. {len(vs.docstore)} chunks indexed.")


with right:
    st.subheader("Ask a Question")
    question = st.text_input("Your question:")
    show_sources = st.checkbox("Show retrieved chunks", value=True)

    if "llm" not in st.session_state:
        with st.spinner("Loading LLM…"):
            st.session_state["llm"] = load_llm(model_id, temperature, max_new_tokens)

    ask = st.button("Get Answer")
    if ask:
        if "vectorstore" not in st.session_state:
            st.warning("Please build the knowledge base first (left panel).")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            vs = st.session_state["vectorstore"]
            llm = st.session_state["llm"]
            retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            chain = make_rag_chain(retriever, llm)
            with st.spinner("Thinking…"):
                answer = chain.invoke(question)

            st.markdown("### Answer")
            st.markdown(answer)  # markdown keeps formatting

            if show_sources:
                st.markdown("### Retrieved Chunks")
                docs = retriever.get_relevant_documents(question)
                for i, d in enumerate(docs, start=1):
                    with st.expander(f"Chunk [{i}]"):
                        st.write(d.page_content)
