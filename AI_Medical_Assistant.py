from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate 
from langchain_classic.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv
import os 

load_dotenv()

api_key = os.getenv("GoogleAPI")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key= api_key)

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = "vector_db"

llm_prompt='''

    You are an intelligent PDF assistant developed by Muhammad Hassan.

    Answer user questions clearly and precisely.
    Provide detailed answers only when the user explicitly asks for details.

    If the question is outside the given context, respond politely:
    "Sorry, I don’t have enough information to answer that."

    Context:
    {context}

    Question:
    {question}

    '''

# def load_pdf(pdfs):
#     documents = []
#     for pdf in pdfs:
#         loader = PyPDFLoader(pdf)
#         documents.extend(loader.load())
#     return documents
import tempfile

def load_pdf(pdfs):
    documents = []

    for pdf in pdfs:
        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents.extend(loader.load())

    return documents

def get_chunks(documents):
    chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunked_document = chunks.split_documents(documents)
    return chunked_document

def create_vector_db(chunked_document, embeddings):
    vector_store = FAISS.from_documents(chunked_document, embeddings)
    vector_store.save_local(vector_db)

def get_conversational_chain():
    prompt = PromptTemplate(template=llm_prompt, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=None,  # you will pass documents manually in your code
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

def users_input(user_input):
    new_db = FAISS.load_local(
        vector_db,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = new_db.as_retriever(search_kwargs={"k": 4})
    chain = get_conversational_chain()

    response = chain({"query": user_input})

    st.write(response["result"])




#--------------------------------------------------------------------------


st.set_page_config(
    page_title="📄 PDF RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PDF Information Extractor")
st.markdown("Upload multiple PDFs and ask questions using AI.")

# Sidebar — PDF Upload
with st.sidebar:
    st.header("📂 Upload PDFs")
    pdf_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("📌 Process PDFs"):
        if not pdf_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                try:
                    docs = load_pdf(pdf_files)
                    chunks = get_chunks(docs)
                    create_vector_db(chunks, embeddings)
                    st.success("✅ PDFs processed and vector database created.")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

# Main Chat Section
st.subheader("💬 Ask a Question")
user_question = st.text_input("Type your question and press Enter")

if user_question:
    if not os.path.exists(vector_db):
        st.warning("Please upload and process PDFs first.")
    else:
        with st.spinner("Searching for an answer..."):
            try:
                users_input(user_question)
            except Exception as e:
                st.error(f"Error: {e}")
