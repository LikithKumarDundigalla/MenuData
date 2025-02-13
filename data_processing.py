import pandas as pd
from io import StringIO
from typing import List, Dict

import streamlit as st

# ------------------------------------------------------------------------------
# Load CSV file
# ------------------------------------------------------------------------------
@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")))


# ------------------------------------------------------------------------------
# Chunking Function
# ------------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


# ------------------------------------------------------------------------------
# Process Data into Document Chunks
# ------------------------------------------------------------------------------
@st.cache_data
def process_data(df: pd.DataFrame) -> List[Dict]:
    documents = []
    grouped = df.groupby(['restaurant_name', 'item_id'])
    for (restaurant, item_id), group in grouped:
        base = group.iloc[0]
        text = (
            f"Restaurant: {base['restaurant_name']}\n"
            f"Address: {base['address1']}, {base['city']}, {base['state']} {base['zip_code']}, {base['country']}\n"
            f"Menu Category: {base['menu_category']}\n"
            f"Menu Item: {base['menu_item']}\n"
            f"Description: {base['menu_description']}\n"
            f"Categories: {base['categories']}\n"
            f"Rating: {base['rating']} (Reviews: {base['review_count']}), Price: {base['price']}\n"
            "Ingredients:\n"
        )
        for _, row in group.iterrows():
            text += f"- {row['ingredient_name']} (confidence: {row['confidence']}) [Item ID: {row['item_id']}]\n"
        documents.append({
            "id": f"{restaurant}_{item_id}",
            "text": text,
            "metadata": {
                "restaurant": restaurant,
                "item_id": item_id,
                "city": base['city'],
                "state": base['state'],
                "rating": base['rating'],
                "price": base['price']
            }
        })

    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['id']}_chunk{i}",
                "text": chunk,
                "metadata": doc["metadata"]
            })
    return all_chunks
