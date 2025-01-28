import streamlit as st
import pandas as pd
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import os

# Load user data from CSV
def load_user_data(file_path):
    return pd.read_csv(file_path)

# Authenticate user
def authenticate_user(username, password, user_data):
    user_record = user_data[(user_data["username"] == username) & (user_data["password"] == password)]
    if not user_record.empty:
        return user_record.iloc[0]
    return None

# Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Initialize RAG components
def initialize_rag_model():
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    return question_encoder, context_encoder, question_tokenizer, context_tokenizer

# Perform RAG-based retrieval
def retrieve_answer(question, context, question_encoder, context_encoder, question_tokenizer, context_tokenizer):
    question_embedding = question_encoder(**question_tokenizer(question, return_tensors="pt"))["pooler_output"]
    context_embeddings = context_encoder(**context_tokenizer(context, return_tensors="pt", padding=True, truncation=True))["pooler_output"]
    similarity_scores = util.pytorch_cos_sim(question_embedding, context_embeddings)
    best_match_idx = similarity_scores.argmax().item()
    return context[best_match_idx]

# Streamlit App
def main():
    # Load user data
    user_data = load_user_data("user_data.csv")  # CSV file containing username, password, and model

    st.title("RAG-Based PDF Retriever")

    # Authentication
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")

    if st.button("Login"):
        user_record = authenticate_user(username, password, user_data)
        if user_record is not None:
            st.success(f"Welcome, {user_record['username']}!")
            
            # Check model match
            model_from_csv = user_record["model"]
            user_model_confirmation = st.radio(
                f"Are you using the {model_from_csv} model as per our records?",
                ["Yes", "No"]
            )

            if user_model_confirmation == "Yes":
                # Access PDF
                pdf_path = st.text_input("Enter the path of the PDF file on your laptop")
                if os.path.exists(pdf_path) and pdf_path.endswith(".pdf"):
                    st.success("PDF found! Extracting content...")
                    pdf_text = extract_text_from_pdf(pdf_path)

                    # Initialize RAG model
                    question_encoder, context_encoder, question_tokenizer, context_tokenizer = initialize_rag_model()

                    # User query for the PDF content
                    question = st.text_input("Ask a question about the PDF content:")
                    if question:
                        context = pdf_text.split(". ")  # Split PDF content into sentences for context
                        answer = retrieve_answer(
                            question, context, question_encoder, context_encoder, question_tokenizer, context_tokenizer
                        )
                        st.subheader("Answer:")
                        st.write(answer)
                else:
                    st.error("Invalid PDF file path. Please try again.")
            else:
                st.warning("You must confirm your model matches our records to proceed.")
        else:
            st.error("Invalid username or password. Please try again.")

if __name__ == "__main__":
    main()
