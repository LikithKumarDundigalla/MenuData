import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def build_faiss_index(all_chunks, wiki_data, news_data: List[Dict]):
    embedder = get_embedder()
    combined_chunks = all_chunks + wiki_data + news_data
    all_texts = [chunk["text"] for chunk in combined_chunks]
    embeddings = embedder.encode(all_texts)
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dimension)
    nlist = 100
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    metadata_store = {}
    for i, chunk in enumerate(combined_chunks):
        metadata_store[i] = chunk

    return index, metadata_store
