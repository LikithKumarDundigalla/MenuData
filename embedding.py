import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
from typing import List, Dict
import warnings
import os
import json
import numpy as np

warnings.filterwarnings('ignore')


@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def build_faiss_index(all_chunks, wiki_data, news_data: List[Dict]):
    index_file = "faiss_index.index"
    metadata_file = "metadata_store.json"


    if os.path.exists(index_file) and os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Empty JSON file")
                metadata_store_json = json.loads(content)
            metadata_store = {int(k): v for k, v in metadata_store_json.items()}
            index = faiss.read_index(index_file)
            return index, metadata_store
        except (json.JSONDecodeError, ValueError) as e:
            # Log the error if needed and continue to rebuild the index.
            print(f"Failed to load metadata from JSON: {e}. Rebuilding the index.")


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


    faiss.write_index(index, index_file)
    with open(metadata_file, "w") as f:
        json.dump({str(k): v for k, v in metadata_store.items()}, f,
                  default=lambda o: int(o) if isinstance(o, np.int64) else o)

    return index, metadata_store
