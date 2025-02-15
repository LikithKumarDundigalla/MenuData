import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from typing import List, Dict
from data_processing import fetch_wikipedia_data, fetch_news_data
import warnings
warnings.filterwarnings('ignore')


@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def build_faiss_index(all_chunks: List[Dict]):
    embedder = get_embedder()
    unique_ingredients = set()
    for chunk in all_chunks:
        if "ingredients" in chunk["metadata"]:
            unique_ingredients.update(chunk["metadata"]["ingredients"].split(", "))

    wiki_chunks = fetch_wikipedia_data(list(unique_ingredients))
    news_chunks = fetch_news_data(list(unique_ingredients))
    combined_chunks = all_chunks + wiki_chunks
    #all_texts = [chunk["text"] for chunk in all_chunks] + [chunk["text"] for chunk in wiki_chunks]
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
