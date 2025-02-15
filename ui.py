import streamlit as st
from data_processing import load_csv, process_data, fetch_wikipedia_data
from embedding import build_faiss_index
from agent import RetrievalAgent
import warnings

warnings.filterwarnings('ignore')

def run():
    st.title("Restaurant Expert Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "response" not in st.session_state:
        st.session_state.response = ""

    uploaded_file = st.file_uploader("Upload Restaurant Data (CSV)", type=["csv"])

    if uploaded_file and st.session_state.agent is None:
        df = load_csv(uploaded_file)
        all_chunks = process_data(df)

        unique_ingredients = set()
        for chunk in all_chunks:
            if "ingredients" in chunk["metadata"]:
                unique_ingredients.update(chunk["metadata"]["ingredients"].split(", "))

        wiki_chunks = fetch_wikipedia_data(list(unique_ingredients))

        # Build FAISS Index (Includes Wikipedia + Restaurant Data)
        index, metadata_store = build_faiss_index(all_chunks + wiki_chunks)

        # Initialize Retrieval Agent
        st.session_state.agent = RetrievalAgent(index, metadata_store)

    user_query = st.text_input("Enter your query:")

    def process_query():
        if user_query and st.session_state.agent:
            st.session_state.response = st.session_state.agent.handle_query(user_query)
            st.session_state.chat_history.append({"user": user_query, "bot": st.session_state.response})

    st.button("Submit Query", on_click=process_query)

    if st.session_state.response:
        st.write(st.session_state.response)

if __name__ == "__main__":
    run()
