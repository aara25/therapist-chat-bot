import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import os,re

# Load API Key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Streamlit UI
st.title("Therapist Chatbot")

# 1️⃣ Load Local PDF File
PDF_PATHS = ["C:\\Users\\vc\\Desktop\\Projects\\Chat Bot using pdf\\stages of counselling.pdf",
            "C:\\Users\\vc\\Desktop\\Projects\\Chat Bot using pdf\\Chapter1IntroductiontoSpeechandLanguageTherapy.pdf",
            "C:\\Users\\vc\\Desktop\\Projects\\Chat Bot using pdf\\chapter16.pdf",
]

def extract_text_from_pdfs(file_paths):
    all_text = ""
    for pdf_path in file_paths:
        if os.path.exists(pdf_path):  # Ensure file exists
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text("text") for page in doc])
            all_text += f"\n\n### {os.path.basename(pdf_path)} ###\n{text}"  # Store text with filename for context
        else:
            st.warning(f"File not found: {pdf_path}")  # Warn if file doesn't exist
    return all_text

if "db" not in st.session_state:
    st.session_state.db = None

if st.session_state.db == None:
    with st.spinner("Loading..."):
        doc =  extract_text_from_pdfs(PDF_PATHS)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_text(doc)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)
        st.session_state.db = FAISS.from_texts(chunks, embeddings)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

    
def get_response(query, chat_history):
    retriever = st.session_state.db.as_retriever()
    docs = retriever.get_relevant_documents(query)

    if docs:
        #Use relevant context from the PDF
        context = "\n\n".join([d.page_content for d in docs[:3]])
        prompt_text = (
            "You are a personal therapist. Answer the following questions considering the"
            "Previous conversation:\n{chat_history}\n\n"
            "Answer based on this context:\n{context}\n\n"
            "Question: {user_question}"
        )
    else:
        # No relevant data → Answer directly using LLM
        context = "No relevant information from the document."
        prompt_text = (
            "You are a personal therapist. Answer the following questions considering the"
            "Previous conversation:\n{chat_history}\n\n"
            "The document has no relevant information. Answer based on general knowledge.\n\n"
            "Question: {user_question}"
        )

    # Define Prompt Template
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "user_question"],
        template=prompt_text
    )

    # LLM Chain with Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", google_api_key=GEMINI_API_KEY)
    chain = prompt | llm | StrOutputParser()

    # Stream response in chunks
    response_generator = chain.stream({
        "chat_history": chat_history,
        "context": context,
        "user_question": query,
    })

    if response_generator is None:
        return iter([])
    return response_generator  # Return generator for streaming


# 5️⃣ Chat Input
query = st.chat_input("Ask something about Therapy...")

if query:
    st.session_state.chat_history.append(HumanMessage(query))
    with st.chat_message("Human"):
        st.markdown(query)

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(query, st.session_state.chat_history))

    if isinstance(ai_response, dict):
        content = ai_response.get("content", "")
    elif isinstance(ai_response, str):
        content = ai_response  # If it's already a string, just use it.

    st.session_state.chat_history.append(AIMessage(content))
