import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ---------- Functions ----------

from urllib.parse import urlparse, parse_qs

def get_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip("/")
    return None

def fetch_transcript(video_id: str) -> str:
    """Fetch transcript from YouTube video."""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list_object = ytt_api.list(video_id)
        transcript_obj = transcript_list_object.find_transcript(['en', 'de']) 
        transcript_list = transcript_obj.fetch() ##print(transcript_list) # Flatten it to plain text
        transcript = " ".join(chunk.text for chunk in transcript_list) ##print(transcript)
        return transcript
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        return ""
    except Exception as e:
        st.error(f"Transcript error: {e}")
        return ""

def build_vectorstore(transcript: str):
    """Create FAISS vector store from transcript."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def build_chain(vector_store):
    """Build main RAG chain."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        huggingfacehub_api_token="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",   # safer than hardcoding
        task="text-generation"
    )
    model = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    parser = StrOutputParser()

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | model | parser
    return main_chain

# ---------- Streamlit UI ----------
st.set_page_config(page_title="YouTube Chatbot", page_icon="🤖", layout="wide")

st.title("🎥 YouTube Video Chatbot")
st.write("Ask questions about any YouTube video using its transcript!")

# Input for YouTube URL
youtube_url = st.text_input("Enter YouTube video URL:")

if youtube_url:
    video_id = get_video_id(youtube_url)
    st.write(f"Extracted Video ID: {video_id}")
    if not video_id:
        st.error("Invalid YouTube URL.")
    else:
        with st.spinner("Fetching transcript..."):
            transcript = fetch_transcript(video_id)
            st.write("Transcript fetched successfully!" if transcript else "No transcript available.")

        if transcript:
            with st.spinner("Building vector store..."):
                vector_store = build_vectorstore(transcript)
                chain = build_chain(vector_store)

            st.success("✅ Ready! Ask anything about the video.")

            # Chat interface
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_query = st.text_input("Ask a question about the video:")

            if st.button("Submit") and user_query:
                with st.spinner("Thinking..."):
                    answer = chain.invoke(user_query)

                # Save history
                st.session_state.chat_history.append(("You", user_query))
                st.session_state.chat_history.append(("Bot", answer))

            # Display chat history
            if st.session_state.chat_history:
                st.subheader("💬 Chat History")
                for speaker, text in st.session_state.chat_history:
                    st.markdown(f"**{speaker}:** {text}")
