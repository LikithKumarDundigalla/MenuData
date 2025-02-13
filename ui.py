import streamlit as st
from data_processing import load_csv, process_data
from embedding import get_embedder, build_faiss_index
from llm_api import generate_answer

def run():
    """ Main function to run the Streamlit UI. """
    st.title("Restaurant Expert Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Upload the Menu Data CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload the CSV file.")
        st.stop()

    df = load_csv(uploaded_file)
    st.write("### Data Sample")
    st.dataframe(df.head())

    all_chunks = process_data(df)
    index, metadata_store = build_faiss_index(all_chunks)
    embedder = get_embedder()

    st.markdown("---")
    st.write("### Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")

    user_query = st.text_input("Enter your query:")
    if st.button("Submit Query") and user_query:
        answer = generate_answer(user_query, index, metadata_store, embedder)
        st.session_state.chat_history.append({"user": user_query, "bot": answer})

if __name__ == "__main__":
    run()
