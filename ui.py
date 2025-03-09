import streamlit as st
import os
import json
import faiss
from agent import RetrievalAgent
import warnings

warnings.filterwarnings('ignore')

def run():
    st.title("Restaurant Expert Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None

    index_file = "faiss_index.index"
    metadata_file = "metadata_store.json"

    # Load the persisted FAISS index and metadata.
    if st.session_state.agent is None:
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            index = faiss.read_index(index_file)
            with open(metadata_file, "r") as f:
                metadata_store_json = json.load(f)
            metadata_store = {int(k): v for k, v in metadata_store_json.items()}
            st.session_state.agent = RetrievalAgent(index, metadata_store)
        else:
            st.info("No persisted index found. Please run the embedding process to generate and save the FAISS index and metadata.")

    # Display existing chat messages.
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("bot"):
            st.write(chat["bot"])

    # Chat input for user queries.
    user_query = st.chat_input("Ask me anything about restaurants!")

    if user_query and st.session_state.agent:
        response = st.session_state.agent.handle_query(user_query)
        st.session_state.chat_history.append({"user": user_query, "bot": response})
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("bot"):
            st.write(response)

if __name__ == "__main__":
    run()
