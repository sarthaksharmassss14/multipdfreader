import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")


def detect_document_type(text):
    lower_text = text.lower()
    if "subject" in lower_text and "marks" in lower_text:
        return "marksheet"
    elif "skills" in lower_text or "experience" in lower_text:
        return "resume"
    else:
        return "unknown"

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

def get_conversational_chain(doc_type):
    prompt_template = """Analyze this pdf and answer only as per the pdf and make no answers on your own.
Context:
{context}

Question:
{question}

Answer:"""

    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain



def user_input(user_question, doc_type):
    if not os.path.exists("faiss_index"):
        st.error("Please upload PDFs first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    chain = get_conversational_chain(doc_type)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("Ask questions from PDF 📄")

    
   with st.sidebar:
    st.title("📄 Upload PDFs")
    
    st.markdown("""
        <div style="border:2px dashed #aaa; padding:15px; border-radius:10px; text-align:center; background-color:#f9f9f9;">
            <strong>📱 Mobile Tip:</strong><br>
            Tap the box below to upload your PDF file.<br>
            Use Chrome or Safari if upload doesn't work.
        </div>
    """, unsafe_allow_html=True)

    pdf_docs = st.file_uploader("📤 Tap here to upload", type=['pdf'], accept_multiple_files=True)

    if st.button("Submit & Process") and pdf_docs:
        with st.spinner("Processing..."):
            text = get_pdf_text(pdf_docs)
            doc_type = detect_document_type(text)
            st.session_state.doc_type = doc_type  
            chunks = get_text_chunks(text)
            get_vector_store(chunks)
            st.success("✅ PDFs processed!")


   
    if "doc_type" in st.session_state:
        user_question = st.text_input("Ask a question from the uploaded PDFs:")
        if user_question:
            user_input(user_question, st.session_state.doc_type)
    else:
        st.warning("Please upload and process PDFs first.")

if __name__ == "__main__":
    main()
