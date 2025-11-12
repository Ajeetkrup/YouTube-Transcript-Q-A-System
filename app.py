import re
import os
from pathlib import Path
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


def get_youtube_transcript(video_url):
    """
    Pulls the transcript from a YouTube video given its URL.
    
    Args:
        video_url (str): YouTube video URL (supports various formats)
        
    Returns:
        str: Full transcript text as a single string
        
    Raises:
        ValueError: If the URL is invalid or video ID cannot be extracted
        TranscriptsDisabled: If transcripts are disabled for the video
        NoTranscriptFound: If no transcript is available
        VideoUnavailable: If the video is unavailable
    """
    video_id = extract_video_id(video_url)
    
    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {video_url}")
    
    try:
        # FIX: Use the new API (v1.0+) - instantiate and use .fetch()
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
        
        # Extract text from the FetchedTranscript object
        transcript_text = ' '.join([snippet.text for snippet in fetched_transcript])
        return transcript_text
    except TranscriptsDisabled:
        raise TranscriptsDisabled(f"Transcripts are disabled for video: {video_id}")
    except NoTranscriptFound:
        raise NoTranscriptFound(f"No transcript found for video: {video_id}")
    except VideoUnavailable:
        raise VideoUnavailable(f"Video is unavailable: {video_id}")


def extract_video_id(url):
    """
    Extracts video ID from various YouTube URL formats.
    
    Args:
        url (str): YouTube URL
        
    Returns:
        str: Video ID or None if not found
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def initialize_groq():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=None,
        max_retries=2,
        groq_api_key=groq_api_key
    )
    return llm


def test_groq_queries():
    llm = initialize_groq()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{text}")
    ])
    
    chain = prompt | llm
    
    test_queries = [
        "Explain the importance of low latency in large language models.",
        "Translate 'I love programming' into French.",
        "Write a haiku about the moon.",
        "What are the key differences between Python and JavaScript?",
        "Summarize the concept of machine learning in one paragraph."
    ]
    
    print("=" * 80)
    print("Testing Groq with different queries:")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}]")
        print(f"Question: {query}")
        print("-" * 80)
        try:
            response = chain.invoke({"text": query})
            print(f"Answer: {response.content}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print("=" * 80)


def chunk_transcript(transcript_text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(transcript_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents


def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def store_in_chromadb(documents, collection_name="youtube_transcripts", persist_directory="./chroma_db"):
    embeddings = get_embedding_function()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    return vectorstore


def load_vectorstore(collection_name="youtube_transcripts", persist_directory="./chroma_db"):
    embeddings = get_embedding_function()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore


def create_retriever(vectorstore, k=4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever


def answer_query(user_query, retriever, llm, k=4):
    retrieved_docs = retriever.invoke(user_query)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context from YouTube video transcripts. 
        Use only the information from the context to answer the question. If the question is out of context or the context doesn't contain enough information to answer the question, 
        you must respond with "I don't know the answer" and do not make up any information. Be factual and accurate in your responses."""),
        ("human", """Context from video transcript:
{context}

Question: {question}

Answer based on the context above:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": user_query})
    return response.content


def test_chromadb_retriever():
    video_url = "https://www.youtube.com/watch?v=15_pppse4fY"
    
    print("=" * 80)
    print("Testing ChromaDB with YouTube Transcript")
    print("=" * 80)
    
    print("\n[Step 1] Fetching YouTube transcript...")
    try:
        transcript = get_youtube_transcript(video_url)
        print(f"✓ Transcript fetched successfully ({len(transcript)} characters)")
    except Exception as e:
        print(f"✗ Error fetching transcript: {str(e)}")
        return
    
    print("\n[Step 2] Chunking transcript...")
    try:
        documents = chunk_transcript(transcript)
        print(f"✓ Transcript chunked into {len(documents)} documents")
        print(f"  Sample chunk (first 200 chars): {documents[0].page_content[:200]}...")
    except Exception as e:
        print(f"✗ Error chunking transcript: {str(e)}")
        return
    
    print("\n[Step 3] Storing chunks in ChromaDB...")
    try:
        vectorstore = store_in_chromadb(documents)
        print("✓ Chunks stored in ChromaDB successfully")
    except Exception as e:
        print(f"✗ Error storing in ChromaDB: {str(e)}")
        return
    
    print("\n[Step 4] Creating retriever...")
    try:
        retriever = create_retriever(vectorstore, k=3)
        print("✓ Retriever created successfully")
    except Exception as e:
        print(f"✗ Error creating retriever: {str(e)}")
        return
    
    print("\n[Step 5] Testing retriever with sample queries...")
    test_queries = [
        "What is the main topic?",
        "What are the key points?",
        "Summarize the content"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 80)
        try:
            retrieved_docs = retriever.invoke(query)
            print(f"Retrieved {len(retrieved_docs)} documents:")
            for j, doc in enumerate(retrieved_docs, 1):
                print(f"\n  Document {j} ({len(doc.page_content)} chars):")
                print(f"  {doc.page_content[:300]}...")
        except Exception as e:
            print(f"✗ Error retrieving: {str(e)}")
        print("=" * 80)


def test_rag_system():
    video_url = "https://www.youtube.com/watch?v=15_pppse4fY"
    
    print("=" * 80)
    print("Testing RAG System: Query -> Embedding -> Similarity Search -> Answer Generation")
    print("=" * 80)
    
    print("\n[Step 1] Fetching YouTube transcript...")
    try:
        transcript = get_youtube_transcript(video_url)
        print(f"✓ Transcript fetched successfully ({len(transcript)} characters)")
    except Exception as e:
        print(f"✗ Error fetching transcript: {str(e)}")
        return
    
    print("\n[Step 2] Chunking transcript...")
    try:
        documents = chunk_transcript(transcript)
        print(f"✓ Transcript chunked into {len(documents)} documents")
    except Exception as e:
        print(f"✗ Error chunking transcript: {str(e)}")
        return
    
    print("\n[Step 3] Storing chunks in ChromaDB...")
    try:
        vectorstore = store_in_chromadb(documents)
        print("✓ Chunks stored in ChromaDB successfully")
    except Exception as e:
        print(f"✗ Error storing in ChromaDB: {str(e)}")
        return
    
    print("\n[Step 4] Creating retriever...")
    try:
        retriever = create_retriever(vectorstore, k=4)
        print("✓ Retriever created successfully")
    except Exception as e:
        print(f"✗ Error creating retriever: {str(e)}")
        return
    
    print("\n[Step 5] Initializing LLM...")
    try:
        llm = initialize_groq()
        print("✓ LLM initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing LLM: {str(e)}")
        return
    
    print("\n[Step 6] Testing RAG with user queries...")
    test_queries = [
        "who is the president of India?",
        "what is agentic AI?",
        "what is the difference between AI and Agentic AI?",
        "difference between AI Agents and Agentic AI?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 80)
        try:
            retrieved_docs = retriever.invoke(query)
            print(f"✓ Retrieved {len(retrieved_docs)} relevant documents")
            
            answer = answer_query(query, retriever, llm)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"✗ Error: {str(e)}")
        print("=" * 80)


def query_existing_vectorstore(user_query, collection_name="youtube_transcripts", persist_directory="./chroma_db", k=4):
    print(f"\n[Query] {user_query}")
    print("-" * 80)
    
    try:
        print("Loading vectorstore...")
        vectorstore = load_vectorstore(collection_name, persist_directory)
        print("✓ Vectorstore loaded")
        
        print("Creating retriever...")
        retriever = create_retriever(vectorstore, k=k)
        print("✓ Retriever created")
        
        print("Initializing LLM...")
        llm = initialize_groq()
        print("✓ LLM initialized")
        
        print("Performing similarity search and generating answer...")
        retrieved_docs = retriever.invoke(user_query)
        print(f"✓ Retrieved {len(retrieved_docs)} relevant documents")
        
        answer = answer_query(user_query, retriever, llm, k=k)
        print(f"\nAnswer:\n{answer}")
        print("=" * 80)
        
        return answer
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


if __name__ == "__main__":
    test_rag_system()