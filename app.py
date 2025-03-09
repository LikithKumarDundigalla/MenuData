import streamlit as st
import ui
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    st.set_page_config(page_title="Restaurant Expert Chatbot", layout="wide")
    ui.run()
