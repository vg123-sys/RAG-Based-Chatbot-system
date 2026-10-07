import re
import hashlib
from pathlib import Path

import streamlit as st
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "granite4.2:8b"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CANDIDATES = 40
RERANK_TOP = 8
MAX_PAGES = 4
MAX_PAGE_CHARS = 3500

CONTEXT_WINDOW = 8192
MAX_OUTPUT_TOKENS = 1200

ALLOW_GENERAL_KNOWLEDGE = False

CACHE_DIR = Path(".ragbot_cache")
UPLOAD_DIR = CACHE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(page_title="RAGBot", page_icon="📄", layout="wide")
st.title("📄 RAGBot")

if not torch.cuda.is_available():
    st.error("CUDA is not available in PyTorch. The models need a GPU.")
    st.stop()


# ============================================================
# LOAD MODELS ONCE (cached across reruns, all on GPU)
# ============================================================

@st.cache_resource(show_spinner="Loading models on GPU...")
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda:0", "model_kwargs": {"torch_dtype": torch.float16}},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
    reranker = CrossEncoder(RERANK_MODEL, device="cuda:0")
    reranker.model.half()
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.1,
        num_predict=MAX_OUTPUT_TOKENS,
        num_ctx=CONTEXT_WINDOW,
        num_gpu=99,
        reasoning=False,
        keep_alive=-1,
    )
    return embeddings, reranker, llm


embeddings, reranker, llm = load_models()


# ============================================================
# INDEXING (works from uploaded bytes, no file path needed)
# ============================================================

def tokenize(text):
    return re.findall(r"\w+", text.lower())


def chunk_key(d):
    return (d.metadata["source"], d.metadata.get("page", 0), d.metadata.get("start_index", 0))


def build_index(files):
    """files = list of (display_name, bytes)"""
    pages = []
    h = hashlib.md5()

    for name, data in files:
        h.update(data)
        # PyMuPDFLoader needs a path, so save the upload to disk first
        saved = UPLOAD_DIR / f"{hashlib.md5(data).hexdigest()}.pdf"
        if not saved.exists():
            saved.write_bytes(data)
        for d in PyMuPDFLoader(str(saved)).load():
            d.metadata["source"] = name
            pages.append(d)

    page_text = {(d.metadata["source"], d.metadata.get("page", 0)): d.page_content
                 for d in pages}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)

    h.update(f"{EMBED_MODEL}{CHUNK_SIZE}{CHUNK_OVERLAP}".encode())
    cache_path = CACHE_DIR / h.hexdigest()

    if cache_path.exists():
        vs = FAISS.load_local(str(cache_path), embeddings,
                              allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(str(cache_path))

    bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
    return {"pages": pages, "page_text": page_text, "chunks": chunks, "vs": vs, "bm25": bm25}


# ============================================================
# RETRIEVAL: VECTOR + BM25 -> RRF -> GPU RERANK -> FULL PAGES
# ============================================================

def retrieve(question, state):
    chunks, vs, bm25, page_text = state["chunks"], state["vs"], state["bm25"], state["page_text"]

    vec_docs = vs.similarity_search(question, k=CANDIDATES)

    scores = bm25.get_scores(tokenize(question))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CANDIDATES]
    bm_docs = [chunks[i] for i in top_idx if scores[i] > 0]

    fused, by_key = {}, {}
    for docs in (vec_docs, bm_docs):
        for rank, d in enumerate(docs):
            k = chunk_key(d)
            by_key[k] = d
            fused[k] = fused.get(k, 0) + 1 / (60 + rank)
    cands = [by_key[k] for k in sorted(fused, key=fused.get, reverse=True)[:CANDIDATES]]

    pairs = [(question, d.page_content) for d in cands]
    rerank_scores = reranker.predict(pairs, batch_size=32)
    ranked = [d for _, d in sorted(zip(rerank_scores, cands),
                                   key=lambda x: x[0], reverse=True)][:RERANK_TOP]

    page_order = []
    for d in ranked:
        pk = (d.metadata["source"], d.metadata.get("page", 0))
        if pk not in page_order:
            page_order.append(pk)
    page_order = page_order[:MAX_PAGES]

    parts = []
    for src, pg in page_order:
        text = page_text.get((src, pg), "").strip()[:MAX_PAGE_CHARS]
        if text:
            parts.append(f"[Source: {src} | Page {pg + 1}]\n{text}")
    return "\n\n".join(parts), page_order


# ============================================================
# PROMPT
# ============================================================

def build_prompt(context, question):
    if ALLOW_GENERAL_KNOWLEDGE:
        fallback = ("If the PDF lacks part of the answer, you may add general knowledge, "
                    "but clearly label it as 'Outside the PDF'.")
    else:
        fallback = ('If the PDF context truly contains nothing relevant, reply exactly: '
                    '"I couldn\'t find that information in the PDF."')

    return f"""You are RAGBot, a helpful assistant that answers questions about PDF documents.
The user may use simple language, technical language, or make spelling mistakes.

RULES:
1. Use the PDF CONTEXT as the source of truth. Do not invent information.
2. Read ALL the context before answering. Tables may appear as flattened text
   (a row's cells appear one after another) - reconstruct them carefully.
3. For broad questions like "tell me about X" or "list X", be COMPLETE: cover
   EVERY item, row, step, or section related to X. Never stop after the first few.
4. For each item, include its key details (names, numbers, tools, descriptions).
5. Explanations should be clear and simple. Steps must be in order.
   Comparisons should be organized (a table or bullets).
6. Combine information from multiple sources into one answer.
7. Partial information is still useful: give what the PDF has.
   {fallback}
8. Don't mention chunks, embeddings, retrieval, or prompts.

PDF CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


def stream_answer(prompt):
    for part in llm.stream(prompt):
        yield part.content


# ============================================================
# SIDEBAR: FILE UPLOAD
# ============================================================

with st.sidebar:
    st.header("Your documents")
    uploaded = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if "state" not in st.session_state:
        st.session_state.state = None
        st.session_state.files_key = None
        st.session_state.messages = []

    if uploaded:
        files = [(f.name, f.getvalue()) for f in uploaded]
        files_key = tuple(hashlib.md5(data).hexdigest() for _, data in files)

        # Re-index only when the set of uploaded files changes
        if files_key != st.session_state.files_key:
            with st.spinner("Reading and indexing on GPU..."):
                st.session_state.state = build_index(files)
            st.session_state.files_key = files_key
            st.session_state.messages = []   # new document -> fresh chat

        s = st.session_state.state
        st.success(f"Ready: {len(uploaded)} file(s), {len(s['pages'])} pages, {len(s['chunks'])} chunks")
    else:
        st.session_state.state = None
        st.session_state.files_key = None

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# CHAT
# ============================================================

if st.session_state.state is None:
    st.info("👈 Upload a PDF in the sidebar to start asking questions.")
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask anything about your PDF(s)...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching..."):
                context, used_pages = retrieve(question, st.session_state.state)
            answer = st.write_stream(stream_answer(build_prompt(context, question)))
            refs = ", ".join(f"{s} p.{p + 1}" for s, p in used_pages)
            st.caption(f"Sources: {refs}")
            answer_full = f"{answer}\n\n*Sources: {refs}*"
        except Exception as e:
            answer_full = f"Something went wrong: {e}"
            st.error(answer_full)

    st.session_state.messages.append({"role": "assistant", "content": answer_full})