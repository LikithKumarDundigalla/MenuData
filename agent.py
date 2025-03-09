import numpy as np
from data_processing import fetch_wikipedia_data, fetch_news_data
from embedding import get_embedder
from llm_api import generate_answer, analyze_query_intent
import warnings
import faiss
import json
import numpy as np

warnings.filterwarnings('ignore')


class RetrievalAgent:
    def __init__(self, index, metadata_store):
        self.index = index
        self.metadata_store = metadata_store
        self.embedder = get_embedder()

    def search_faiss(self, query, top_k=10):
        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vec, dtype=np.float32), top_k)
        retrieved = [self.metadata_store[idx] for idx in indices[0] if idx in self.metadata_store]
        return retrieved if retrieved else None

    def search_external_sources(self, query):
        query_categories = analyze_query_intent(query)
        combined_data = []

        if 'historical' in query_categories:
            combined_data.extend(fetch_wikipedia_data([query]))
        if 'current' in query_categories:
            combined_data.extend(fetch_news_data(query))

        return combined_data if combined_data else [{"text": f"No direct info found for {query}."}]

    def update_faiss(self, new_data):
        new_texts = [chunk["text"] for chunk in new_data]
        new_embeddings = self.embedder.encode(new_texts)

        num_existing_vectors = self.index.ntotal
        self.index.add(np.array(new_embeddings, dtype=np.float32))

        for i, chunk in enumerate(new_data):
            self.metadata_store[num_existing_vectors + i] = chunk


        faiss.write_index(self.index, "faiss_index.index")
        with open("metadata_store.json", "w") as f:
            json.dump({str(k): v for k, v in self.metadata_store.items()}, f,
                      default=lambda o: int(o) if isinstance(o, np.int64) else o)

    def handle_query(self, query):
        retrieved_chunks = self.search_faiss(query)
        if retrieved_chunks:
            return generate_answer(query, self.index, self.metadata_store, self.embedder)
        else:
            external_data = self.search_external_sources(query)
            self.update_faiss(external_data)
            return generate_answer(query, self.index, self.metadata_store, self.embedder)
