import pandas as pd
from io import StringIO
from typing import List, Dict
import wikipedia
import streamlit as st
import requests
import warnings
warnings.filterwarnings('ignore')


@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
   return pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")))

def extract_ingredients(all_chunks):
    """Extracts unique ingredients from restaurant data."""
    unique_ingredients = set()
    for chunk in all_chunks:
        if "ingredients" in chunk["metadata"]:
            unique_ingredients.update(chunk["metadata"]["ingredients"].split(", "))
    return list(unique_ingredients)

def fetch_external_data(all_chunks):
    """Fetch Wikipedia and news data for extracted ingredients."""
    ingredients = extract_ingredients(all_chunks)
    wiki_chunks = fetch_wikipedia_data(ingredients)
    news_chunks = fetch_news_data(ingredients)
    return wiki_chunks, news_chunks

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

@st.cache_data
def fetch_wikipedia_data(item: List[str]) -> List[Dict]:
    wiki_api = wikipedia.set_lang('en')
    wiki_docs = []

    for i in item:
        page = wiki_api.page(i)
        if page.exists():
            wiki_docs.append({
                "id": f"wiki_{i}",
                "text": page.summary,
                "metadata": {"source": "Wikipedia", "ingredient": i}
            })

    return wiki_docs


def fetch_news_data(query):
    api_key = '36076cb472d44740a7cbda95a98c783c'
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])

    news_data = []
    for article in articles:
        news_data.append({
            "id": f"news_{article['title'].replace(' ', '_')}",
            "text": f"{article['title']} - {article['description']}",
            "metadata": {
                "source": "News",
                "url": article['url']
            }
        })
    return news_data
