import streamlit as st
from app import (
    get_youtube_transcript,
    chunk_transcript,
    store_in_chromadb,
    create_retriever,
    initialize_groq,
    answer_query
)

st.set_page_config(page_title="YouTube Transcript Q&A", page_icon="📺", layout="wide")

st.title("📺 YouTube Transcript Q&A System")
st.markdown("Ask questions about YouTube video transcripts using AI")

if "transcript_loaded" not in st.session_state:
    st.session_state.transcript_loaded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "video_url" not in st.session_state:
    st.session_state.video_url = ""

st.divider()

st.header("Step 1: Load Transcript")
video_url = st.text_input(
    "Enter YouTube URL",
    value=st.session_state.video_url,
    placeholder="https://www.youtube.com/watch?v=...",
    help="Paste the YouTube video URL here"
)

if st.button("📥 Download Transcript", type="primary"):
    if not video_url:
        st.error("Please enter a YouTube URL")
    else:
        with st.spinner("Fetching transcript..."):
            try:
                transcript = get_youtube_transcript(video_url)
                st.success(f"✓ Transcript fetched successfully ({len(transcript)} characters)")
                
                with st.spinner("Processing transcript (chunking and creating embeddings)..."):
                    documents = chunk_transcript(transcript)
                    vectorstore = store_in_chromadb(documents)
                    retriever = create_retriever(vectorstore, k=4)
                    llm = initialize_groq()
                    
                    st.session_state.retriever = retriever
                    st.session_state.llm = llm
                    st.session_state.transcript_loaded = True
                    st.session_state.video_url = video_url
                    
                st.success("✓ Transcript processed and ready for queries!")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()

st.header("Step 2: Ask Questions")

if st.session_state.transcript_loaded:
    user_query = st.text_input(
        "Enter your question",
        placeholder="What is the main topic of this video?",
        help="Ask any question about the video content"
    )
    
    if st.button("🔍 Submit Query", type="primary"):
        if not user_query:
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching and generating answer..."):
                try:
                    answer = answer_query(
                        user_query,
                        st.session_state.retriever,
                        st.session_state.llm
                    )
                    
                    st.subheader("Answer:")
                    st.info(answer)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👆 Please load a transcript first before asking questions")
    st.text_input(
        "Enter your question",
        placeholder="Load transcript first...",
        disabled=True
    )

