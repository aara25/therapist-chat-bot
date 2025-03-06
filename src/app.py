import os
import psycopg2
import streamlit as st
from pgvector.psycopg2 import register_vector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.pgvector import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Streamlit UI Config
st.set_page_config(page_title="Therapist Chatbot", page_icon="🧠")
st.title("🧠 Therapist Chatbot")

# Initialize database session
if "db" not in st.session_state:
    st.session_state.db = None

# Function to get database connection
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn

# Function to store chat history in PostgreSQL
def store_chat_history(user_id, user_query, ai_response):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            user_query TEXT,
            ai_response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute(
        "INSERT INTO chat_history (user_id, user_query, ai_response) VALUES (%s, %s, %s)",
        (user_id, user_query, ai_response)
    )
    conn.commit()
    cursor.close()
    conn.close()

# Function to retrieve full chat history from PostgreSQL
def get_chat_history(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_query, ai_response FROM chat_history WHERE user_id = %s ORDER BY timestamp ASC",
        (user_id,)
    )
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    return history

# Function to generate response from LLM
def get_response(query, user_id):
    chat_history = get_chat_history(user_id)  # Fetch entire conversation history
    chat_history_str = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in chat_history])

    retriever = st.session_state.db.as_retriever() if st.session_state.db else None
    docs = retriever.get_relevant_documents(query) if retriever else []
    context = "\n\n".join([d.page_content for d in docs[:3]]) if docs else ""

    # Prompt template
    prompt_text = (
        "You are a personal therapist. Provide therapy-related support and guidance. "
        "Keep responses warm, empathetic, and focused on mental health.\n\n"
        "Past conversation:\n{chat_history}\n\n"
        "Context:\n{context}\n\n"
        "User's question: {user_question}"
    )
    prompt = PromptTemplate(input_variables=["chat_history", "context", "user_question"], template=prompt_text)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", google_api_key=GEMINI_API_KEY)
    chain = prompt | llm | StrOutputParser()
    
    response_generator = chain.stream({
        "chat_history": chat_history_str,
        "context": context,
        "user_question": query,
    })
    
    return response_generator or iter([])

# Function to process uploaded PDF file
def process_pdf(file):
    save_path = os.path.join("temp", file.name)
    os.makedirs("temp", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(file.getbuffer())

    pdf_loader = PyMuPDFLoader(save_path)
    docs = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Store in PostgreSQL using pgvector
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    connection = get_db_connection()
    st.session_state.db = PGVector.from_documents(chunks, embeddings, connection_string=DATABASE_URL, collection_name="therapy_chatbot_embeddings")

    st.success("PDF processed and stored in vector database!")

# Sidebar for uploading PDF file
st.sidebar.header("Upload Therapist Guide (PDF)")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    process_pdf(uploaded_file)

# Chat Interface
user_id = "default_user"  # Replace with actual user authentication if available
chat_history = get_chat_history(user_id)

# Display past chat history within the chat interface
for user_msg, bot_msg in chat_history:  # Show full conversation in order
    with st.chat_message("Human"):
        st.markdown(user_msg)
    with st.chat_message("AI"):
        cleaned_bot_msg = bot_msg.replace("Bot:", "").strip()
        st.markdown(cleaned_bot_msg)

# Chat input
query = st.chat_input("Ask a therapy-related question...")
if query:
    with st.chat_message("Human"):
        st.markdown(query)

    with st.chat_message("AI"):
        ai_response = get_response(query, user_id)  # Generator

        # Stream the response in real-time
        response_content = st.write_stream(ai_response)

    store_chat_history(user_id, query, response_content)
