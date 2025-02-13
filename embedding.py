import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from typing import List, Dict

# ------------------------------------------------------------------------------
# Load SentenceTransformer Model
# ------------------------------------------------------------------------------
@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')


# ------------------------------------------------------------------------------
# Build FAISS Index
# ------------------------------------------------------------------------------
@st.cache_resource
def build_faiss_index(all_chunks: List[Dict]):
    embedder = get_embedder()
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedder.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    metadata_store = {i: all_chunks[i] for i in range(len(all_chunks))}
    return index, metadata_store
