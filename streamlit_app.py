import os
import sys
import streamlit as st
import pickle

# Set up paths
dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(dir, "./bin")))

# Imports
from vectordb import get_vector_db_from_persist_directory
from retrievar import vector_retrievar_with_source, get_context, ensemble_retriever
from llm_generate import get_response_streaming
from langchain_openai import AzureOpenAIEmbeddings

# Config
azure_openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"]
azure_openai_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint
api_version = st.secrets["API_VERSION"]
gen_model = st.secrets["AZURE_OPENAI_GENERATION_MODEL"]

# Load Chunked Data
def load_docs_pickle(path="./data/chunked_rows.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
chunked_docs = load_docs_pickle()

# Vector DB
persist_directory = "./vector_db"
collection_name = "algo_aces"
embedding_model = AzureOpenAIEmbeddings(model=st.secrets["AZURE_OPENAI_EMBEDDING_MODEL"])
db = get_vector_db_from_persist_directory(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_name=collection_name
)

# Page styling
st.set_page_config(page_title="Algo Aces Chatbot", layout="wide")
st.markdown("<h1 style='text-align: center;'>🤖 Algo Aces Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Simple Chatbot which answers user queries!</p>", unsafe_allow_html=True)
st.markdown("---")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("💬 Enter your query..."):

    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"🧑‍💻 **You:** {prompt}")

    # Retrieve context
    with st.spinner("🔎 Retrieving relevant context..."):
        retrieved_docs, sources = ensemble_retriever(chunked_docs, vector_db=db, query=prompt, top_k=10)
        context = get_context(retrieved_docs)

    with st.expander("📄 Context sources used"):
        for source in sources:
            st.markdown(f"- {source}")

    # Stream LLM response
    with st.spinner("💡 Generating response..."):
        res_gen = get_response_streaming(
            primary_model=gen_model,
            api_version=api_version,
            question=prompt,
            context=context
        )

        full_response = []
        with st.chat_message("assistant"):
            for chunk in st.write_stream(res_gen()):
                full_response.append(chunk)

        complete_response = "".join(full_response)
        st.session_state.messages.append({"role": "assistant", "content": complete_response})

    st.markdown("---")