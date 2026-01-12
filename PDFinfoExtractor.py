import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS    
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os 

load_dotenv()
# api_key = st.secrets["APIkey"]   
api_key= os.getenv("GoogleAPI") 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=api_key,  temperature=0.3)

prompt_template='''

    you developed/build by Muhammad Hassan an AI developer.    
    You are an intelligent pdf reader. Your work is to give thr brief answer to the user in a very
    precise way.
    Make sure to provide every single detail about the question given by the user if he is asking 
    for details . 
    Don't be so random.

    Only greet the user if he greets else answer the query given by the user  
    when the user greets like [hi, hello , hey , hello buddy, hey buddy]. Greet him back in a friendly way
    and ask him how was your day or how is he doing making tone friendly and professional.

    If the question is not from the from the given context just answer them in a politely way that  
    "Sorry! I dont have enough information".

    context:{context}
    question:{question}
    '''


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks= text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    vector_store= FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt= PromptTemplate(template= prompt_template, input_variables=["context", "question"] )

    chain= load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    new_db= FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs= new_db.similarity_search(user_question)

    chain= get_conversational_chain()

    response= chain(
        {"input_documents": docs, 
         "question": user_question}, 
         return_only_outputs=True
         )

    print(response)

    st.write("Reply: ", response['output_text'])


# Streamlit UI

st.set_page_config(page_title="📄 PDF Chatbot with Gemini", page_icon="🤖", layout="wide")
st.title("🤖 PDF Information Extractor ")
st.markdown("Upload Multiple PDFs, process them, and chat with your data!")

with st.sidebar:
    st.header("📄 Upload your PDF files")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDFs processed successfully and vector store saved!")
                else:
                    st.warning("❗ No text extracted from uploaded PDFs.")
        else: 
            st.warning("❗ Please upload at least one PDF file.")

user_question = st.text_input("Enter your question:")

if user_question:   
    with st.spinner("Searching for the answer..."):
        try:
            user_input(user_question)
        except Exception as e:
            st.error(f"❗ Error: {e}")