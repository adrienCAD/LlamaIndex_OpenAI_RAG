import os

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import openai

from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, Settings, ServiceContext, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext

# import chromadb for storing embeddings
import chromadb

# import OpenAI API key
openai.api_key =  os.getenv("OPENAI_API_KEY")

# Set the StreamLit page configuration
st.set_page_config(page_title="Chat with the docs in the PDF folder, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Retrieval Augmentation System using provided PDF information and powered by LlamaIndex and OpenAI 🦙🦾")
st.info("[source](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

# Add a sidebar with a dropdown menu
llm_engine = st.sidebar.selectbox("Select LLM Engine", ["gpt-3.5-turbo", "gpt-4", "llama-7b"])

# Print the selected value in the console
print("Selected LLM Engine:", llm_engine)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Chemical Engineering!"}
            ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take one minute."):
        reader = SimpleDirectoryReader(input_dir="../PDFs", recursive=True)
        docs = reader.load_data()

        embed_model = OpenAIEmbedding()
        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

        # OpenAI Settings
        Settings.llm = OpenAI(
                temperature=0.1,
                model=llm_engine,
                # model="gpt-4",
                system_prompt="You are an expert with global expertise, and with extensive knowledge in Chemical Engineering and Sustainability. Many questions will be related to Process Engineering, Process Simulation, Capital Cost Estimation, Front End Engineering Design, or Digital Twins in general. Try to keep your answers technical and based on facts, but if you receive a question outside the context you are familiar with, then use the OpenAI Agent Mode to provide an answer to the best of your ability. Limit feature hallucination as much as possible."
                )
        Settings.embed_model=embed_model
        Settings.node_parser=node_parser

        # STORING THE EMBEDDINGS IN CHROMA DB
        # initialize client, setting path to save data
        db = chromadb.PersistentClient(path="./chroma_db")

        # create collection
        chroma_collection = db.get_or_create_collection("quickstart")

        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Top-k semantic retrieval
        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context
            )

        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
