import streamlit as st
from data_processing import load_csv, process_data, fetch_external_data
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

    uploaded_file = st.file_uploader("Upload Restaurant Data (CSV)", type=["csv"])

    if uploaded_file and st.session_state.agent is None:
        df = load_csv(uploaded_file)
        all_chunks = process_data(df)
        wiki_data, news_data = fetch_external_data(all_chunks)
        index, metadata_store = build_faiss_index(all_chunks, wiki_data, news_data)
        st.session_state.agent = RetrievalAgent(index, metadata_store)

    # Display chat messages
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("bot"):
            st.write(chat["bot"])

    # Chat input for user messages
    user_query = st.chat_input("Ask me anything about restaurants!")

    if user_query and st.session_state.agent:
        response = st.session_state.agent.handle_query(user_query)
        st.session_state.chat_history.append({"user": user_query, "bot": response})

        # Display the new messages immediately
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("bot"):
            st.write(response)

if __name__ == "__main__":
    run()
