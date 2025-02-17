# Restaurant Expert Chatbot

## Overview
The **Restaurant Expert Chatbot** is a Streamlit-based AI chatbot designed to provide users with expert insights about restaurants. It leverages a combination of **local restaurant data**, **Wikipedia information**, and **news sources** to answer user queries. The chatbot employs **FAISS for efficient retrieval** and **an LLM-powered API for response generation**.

## Features
- Upload a **CSV file** containing restaurant data.
- Process restaurant details, menu items, and ingredients.
- Fetch **external information** from Wikipedia and news sources.
- Utilize **FAISS** for fast and efficient restaurant data retrieval.
- Generate intelligent responses using a **language model API**.
- Maintain a chat history within the Streamlit interface.

---
## Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** installed, along with the required dependencies:

### 1. Clone the Repository
```sh
$ git clone https://github.com/your-repo/restaurant-chatbot.git
$ cd restaurant-chatbot
```

### 2. Install Dependencies

Create a virtual environment and install the required dependencies:

```sh
$ python -m venv venv
$ source venv/bin/activate  # On Windows use `venv\Scripts\activate`
$ pip install -r requirements.txt
```
### 3. Run the Chatbot
``` sh
$ streamlit run app.py
```
This will launch the chatbot in your browser.

## Usage 
1. Upload a CSV file containing restaurant data.
2. The chatbot processes the data and builds an FAISS-based retrieval index.
3. Ask questions related to restaurants, menu items, and ingredients.
4. The chatbot retrieves the best-matching results and generates responses.
5. External sources (Wikipedia and News API) are used to enhance responses when needed.

## Project Structure
```
restaurant-chatbot/
├── app.py               # Main Streamlit application
├── ui.py                # Handles UI interactions
├── data_processing.py   # Loads and processes restaurant data
├── embedding.py         # Builds FAISS index and handles embeddings
├── agent.py             # RetrievalAgent for querying indexed data
├── llm_api.py           # Generates chatbot responses via LLM API
├── retrieval.py         # Retrieves relevant text chunks from FAISS
├── requirements.txt     # Dependencies list
└── README.md            # Documentation
```

## API Usage

### 1. Getting Your Mistral.ai API Key
1. Visit to the website [Mistral.ai](https://mistral.ai/en).
2. Locate the option ```Try the API``` and click on it. 
3. Sign up or log in at Mistral.ai
4. Navigate to your account settings or developer dashboard.
5. Find the section for API, and generate a new API key if you don’t have one already.
6. Copy the API key.

The chatbot integrates with an LLM API for generating responses. Ensure your API key is correctly set in ``llm_api.py``:
```
headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}
```
To modify query processing, adjust the prompt construction in ``llm_api.py``.

## Future Improvements

## Additional Future Enhancements

#### 1. User Feedback Loop
- **Rating and Reporting**: Let users rate the chatbot’s responses or flag incorrect/incomplete answers. This feedback can help refine the chatbot’s data and improve your retrieval or LLM prompting over time.

#### 2. Real-Time Data Updates
- **Dynamic Source Monitoring**: For restaurant data that changes frequently (e.g., menus, operating hours), integrate triggers or scheduled jobs to update indexes.  
- **Cache Invalidation**: Consider strategies to invalidate stale embeddings when data changes, ensuring fresh, accurate content.

#### 3. Automated Testing & Monitoring
- **Unit and Integration Tests**: Validate that each module (e.g., data ingestion, embedding, retrieval) functions as intended.  
- **Performance Metrics**: Track query latency, response accuracy, and embedding indexing speed to identify bottlenecks.

#### 4. Multi-language Support 
If your user base spans multiple languages, consider incorporating language detection and translation features for the queries and responses.

#### 5. Authentication & Access Control
If you plan to handle sensitive restaurant or user data, implement authentication (e.g., OAuth2) to limit who can upload data or access certain functionalities.
