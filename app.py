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

import yt_dlp
import os

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv(".env")

# =========================
# GET SECRETS
# =========================

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(
    page_title="Qwen YouTube Chatbot",
    layout="wide"
)

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

    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    return url

# =========================
# FETCH TRANSCRIPT FUNCTION
# =========================

def fetch_transcript(video_id):

    # -------------------------
    # METHOD 1
    # youtube-transcript-api
    # -------------------------

    try:

        api = YouTubeTranscriptApi()

        transcript = api.fetch(video_id)

        full_text = " ".join(
            [item.text for item in transcript]
        )

        return full_text

    except Exception:

        st.warning(
            "youtube-transcript-api blocked. Trying yt-dlp fallback..."
        )

        # -------------------------
        # METHOD 2
        # yt-dlp fallback
        # -------------------------

        try:

            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "quiet": True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:

                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False
                )

            subtitles = info.get("automatic_captions")

            if subtitles is None:
                subtitles = info.get("subtitles")

            if subtitles is None:
                return None

            transcript_text = str(subtitles)

            return transcript_text

        except Exception as yt_error:

            st.error(f"Both methods failed:\n\n{yt_error}")

            return None

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
# SIDEBAR
# =========================

with st.sidebar:

    st.header("Load YouTube Video")

    youtube_url = st.text_input(
        "Paste YouTube URL"
    )

    load_video = st.button(
        "Load Video",
        use_container_width=True
    )

    if st.session_state.vectorstore:
        st.success("Video Loaded ✅")

# =========================
# SHOW OLD CHATS
# =========================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# LOAD VIDEO
# =========================

if load_video:

    with st.spinner("Loading Video..."):

        video_id = extract_video_id(youtube_url)

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

            full_text = fetch_transcript(video_id)

            if full_text is None:

                st.error("Could not fetch transcript")

                st.stop()

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

    with st.chat_message("user"):
        st.markdown(question)

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
    # CHAT HISTORY
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

        Give proper conceptual answers and try to keep things crisp and simple.

        Make sure to give answer in markdown format. 
        Behave like a human with text and sounds good to them 
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
    # SAVE HISTORY
    # =========================

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": response
        }
    )

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