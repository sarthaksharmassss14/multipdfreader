import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os
import json
from pathlib import Path
import uuid


# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# PDF Handling
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

# Chain & Model Setup
def get_conversational_chain(doc_type):
    prompt_template = """Analyze this pdf and answer only as per the pdf and make no answers on your own.
Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Save Q&A
def save_qa_to_json(question, answer, email, file_path):
    qa_entry = {
        "email": email,
        "question": question,
        "answer": answer
    }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if Path(file_path).exists():
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(qa_entry)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


# Answer questions
def user_input(user_question, doc_type):
    if not os.path.exists("faiss_index"):
        st.error("Please upload PDFs first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    chain = get_conversational_chain(doc_type)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    answer = response["output_text"]
    st.write("Reply:", answer)

    # If email is available, save directly to file
    if st.session_state.email:
        save_qa_to_json(user_question, answer, st.session_state.email, st.session_state.json_file)
    else:
        st.session_state.temp_qa_log.append({
            "question": user_question,
            "answer": answer,
            "email": st.session_state.email 
        })

   # Log chat to Mongo only if email is provided
    if st.session_state.email:
        from logger import log_chat
        log_chat(user_input=user_question, bot_response=answer, email=st.session_state.email)




# Main App
def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("Ask questions from PDF 📄")

    # Session variables
    if "temp_qa_log" not in st.session_state:
        st.session_state.temp_qa_log = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "json_file" not in st.session_state:
        st.session_state.json_file = f"qa_logs/qa_{st.session_state.session_id}.json"
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0
    if "email" not in st.session_state:
        st.session_state.email = None
    if "email_prompted" not in st.session_state:
        st.session_state.email_prompted = False

    # PDF Upload
    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                text = get_pdf_text(pdf_docs)
                doc_type = detect_document_type(text)
                st.session_state.doc_type = doc_type
                chunks = get_text_chunks(text)
                get_vector_store(chunks)
                st.success("PDFs processed")

    # Chat Interface
    if "doc_type" in st.session_state:
        user_question = st.text_input("Ask a question from the uploaded PDFs:")

        if user_question:
            st.session_state.message_count += 1

            # Get answer always
            user_input(user_question, st.session_state.doc_type)

            # Prompt for email every 3 messages until provided
            if st.session_state.email is None and st.session_state.message_count % 3 == 0:
                st.session_state.email_prompted = True

        # If it's time to ask for email
        if st.session_state.email_prompted and st.session_state.email is None:
            email_input = st.text_input("🔒 Please enter your email to enable saving your Q&A:")

            if email_input and "@gmail.com" in email_input:
                # Save temp QA log to file after getting email
                for qa in st.session_state.temp_qa_log:
                    save_qa_to_json(qa["question"], qa["answer"], email_input, st.session_state.json_file)
                st.session_state.temp_qa_log = []  # Clear after saving

                st.session_state.email = email_input
                st.session_state.email_prompted = False
                st.success("✅ Email saved. Your Q&A history is saved.")
            elif email_input:
                st.warning("❌ Invalid email. Please enter a valid one.")

    else:
        st.warning("Please upload and process PDFs first.")

    # JSON Download if email given and file exists
    if Path(st.session_state.json_file).exists() and st.session_state.email:
        with open(st.session_state.json_file, "rb") as f:
            st.download_button(
                label="📥 Download Q&A JSON",
                data=f,
                file_name=os.path.basename(st.session_state.json_file),
                mime="application/json"
            )
        st.markdown(f"Saved at: `{st.session_state.json_file}`")



if __name__ == "__main__":
    main()

