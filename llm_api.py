import json
import requests
from typing import List, Dict
from retrieval import retrieve_relevant_chunks
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

MISTRALAI_KEY = st.secrets["MISTRALAI_KEY"]

def construct_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source_info = (
            f"[Source: {chunk['metadata'].get('restaurant', 'external')}, "
            f"ID: {chunk['id']}, City: {chunk['metadata'].get('city')}]"
        )
        context_lines.append(f"{i}. {chunk['text']} {source_info}")
    context_text = "\n".join(context_lines)
    prompt = (
            "You are a restaurant expert AI that integrates internal restaurant data with verified external sources focusing exclusively on San Fransisco. "
            "Below is the context derived from our internal data. Use it to generate a well-reasoned yet concise answer to the user's query, ensuring logical coherence and factual accuracy."
            "Provide inline source references where applicable. Do not reveal any internal reasoning or chain-of-thought; provide only the final answer.\n\n"
            "If there is no relevant context available, generate the most accurate response based on external sources and general knowledge.\n\n"
            "Context:\n" + context_text + "\n\n"
            "User Query: " + query + "\n\n"
            "Answer:"
    )
    return prompt


def generate_answer(query: str, index, metadata_store, embedder) -> str:
    retrieved_chunks = retrieve_relevant_chunks(query, index, metadata_store, embedder)
    prompt_text = construct_prompt(query, retrieved_chunks)

    url = "https://api.mistral.ai/v1/chat/completions"
    payload = json.dumps({
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": "You are a helpful, factual, and precise restaurant data assistant."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 500,
        "stream": False
    })
    headers = {
        'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Authorization': 'Bearer 9wYAB87lJfh0fkuPJuB33j33PXCkZ9Ic',
  'Cookie': '__cf_bm=ZtlKt96_PMs9oRwMM_TG_3_yO4Ux_W7L5xligFswDAo-1739421273-1.0.1.1-hZpK8g_ZnY.Wb9epx5grPcVedplnvcfROHD3EibkIPg181z.Z7SlOcqoG28VyKBDVggpMa0AybXvRV938dhq3w'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        response_data = response.json()
        answer = response_data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, json.JSONDecodeError):
        answer = "Sorry, I couldn't generate a response."

    return answer


def analyze_query_intent(query):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        'Authorization': 'Bearer 9wYAB87lJfh0fkuPJuB33j33PXCkZ9Ic',
        'Content-Type': 'application/json'
    }
    data = {"query": query}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json().get('categories', [])
    else:
        return []