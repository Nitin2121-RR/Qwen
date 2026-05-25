import streamlit as st

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from googleapiclient.discovery import build
from dotenv import load_dotenv

import os

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv(".env")

# Use st.secrets on Streamlit Cloud, fallback to .env locally
def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

# =========================
# STREAMLIT TITLE
# =========================

st.title("Qwen YouTube Chatbot")

# =========================
# YOUTUBE API
# =========================

youtube = build(
    "youtube",
    "v3",
    developerKey=get_secret("YOUTUBE_API_KEY")
)

# =========================
# CACHE EMBEDDINGS
# =========================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =========================
# CACHE LLM
# =========================

@st.cache_resource
def load_llm():

    llm = HuggingFaceEndpoint(

        repo_id="meta-llama/Llama-3.1-8B-Instruct",

        huggingfacehub_api_token=get_secret("HUGGING_FACE"),

        task="conversational",

        temperature=0.5
    )

    return ChatHuggingFace(llm=llm)

# =========================
# LOAD MODELS
# =========================

embeddings = load_embeddings()
model = load_llm()

# =========================
# EXTRACT VIDEO ID
# =========================

def extract_video_id(url):

    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]

    return url

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# =========================
# SHOW OLD CHATS
# =========================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# URL INPUT
# =========================

youtube_url = st.text_input("Enter YouTube URL")

# =========================
# LOAD VIDEO
# =========================

if st.button("Load Video"):

    with st.spinner("Loading Video..."):

        video_id = extract_video_id(youtube_url)

        # Separate DB for every video
        persist_directory = f"db/{video_id}"

        # =========================
        # LOAD SAVED VECTORSTORE
        # =========================

        if os.path.exists(persist_directory):

            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )

            st.session_state.vectorstore = vectorstore

            st.success("Loaded From Saved Database ⚡")

        else:

            # =========================
            # FETCH TRANSCRIPT
            # =========================

            api = YouTubeTranscriptApi()

            transcript = api.fetch(video_id)

            full_text = " ".join(
                [item.text for item in transcript]
            )

            # =========================
            # DOCUMENT
            # =========================

            docs = [
                Document(page_content=full_text)
            ]

            # =========================
            # SPLIT DOCUMENTS
            # =========================

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            split_docs = splitter.split_documents(docs)

            # =========================
            # CREATE VECTORSTORE
            # =========================

            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=persist_directory
            )

            # Save DB
            vectorstore.persist()

            st.session_state.vectorstore = vectorstore

            st.success("New Video Processed & Saved ✅")

# =========================
# CHAT INPUT
# =========================

question = st.chat_input("Ask question from video")

# =========================
# QUESTION ANSWERING
# =========================

if question and st.session_state.vectorstore:

    # =========================
    # SHOW USER MESSAGE
    # =========================

    with st.chat_message("user"):
        st.markdown(question)

    # Save User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # =========================
    # RETRIEVER
    # =========================

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # =========================
    # CREATE CHAT HISTORY
    # =========================

    chat_history = ""

    recent_chats = st.session_state.chat_history[-6:]

    for chat in recent_chats:

        human_question = chat["question"]
        ai_answer = chat["answer"]

        chat_history += f"Human: {human_question}\n"
        chat_history += f"AI: {ai_answer}\n"

    # =========================
    # PROMPT
    # =========================

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful YouTube video assistant.

        Give proper conceptual answer as per the context and previous chat history.
        If possible give a proper answer in bullet points and proper structured way

        Also if it seems to be a normal greeting just great like hey , hello or else 
        thing is asked not related to the context just give a proper answer of it that 
        is not related to the context and give a responce like a human greet.


        Previous Chat History:
        {chat_history}

        Context:
        {context}

        Current Question:
        {question}
        """
    )

    # =========================
    # FORMAT DOCS
    # =========================

    def format_docs(docs):

        return "\n\n".join(
            [doc.page_content for doc in docs]
        )

    # =========================
    # CHAIN
    # =========================

    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x,
            "chat_history": lambda x: chat_history
        }
        | prompt
        | model
        | StrOutputParser()
    )

    # =========================
    # GENERATE RESPONSE
    # =========================

    response = chain.invoke(question)

    # =========================
    # SAVE CHAT HISTORY
    # =========================

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": response
        }
    )

    # =========================
    # SAVE ASSISTANT MESSAGE
    # =========================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    # =========================
    # SHOW RESPONSE
    # =========================

    with st.chat_message("assistant"):
        st.markdown(response)