import numpy as np
from typing import List, Dict

def retrieve_relevant_chunks(query: str, index, metadata_store, embedder, top_k: int = 5) -> List[Dict]:
    query_vec = embedder.encode([query])
    query_vec = np.array(query_vec, dtype=np.float32)
    distances, indices = index.search(query_vec, top_k)
    retrieved = [metadata_store[idx] for idx in indices[0] if idx in metadata_store]
    return retrieved
