import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Load user data
df = pd.read_csv("c:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\user_data.csv")
manual_data = pd.read_csv("C:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\laptop_models.csv")

# Streamlit Interface
st.title("CHATBOT")
st.sidebar.title("Welcome to Galaxy Office Automation Pvt Ltd")
st.sidebar.image(
    "c:\\Users\\Heeta Parmar\\Downloads\\Galaxy.png",
    caption="User Dashboard",
    use_container_width=True,
)

# Session state initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "model_confirmed" not in st.session_state:
    st.session_state.model_confirmed = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# PDF processing
def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() for page in reader.pages)
    return text

def create_retriever(pdf_content):
    embeddings = HuggingFaceEmbeddings(model_name="facebook/dpr-question_encoder-multiset-base")
    vectorstore = FAISS.from_texts([pdf_content], embeddings)
    return vectorstore.as_retriever()

# Login and model selection
def login_and_model_selection():
    st.title("Login")
    username = st.text_input("Enter your username:")
    password = st.text_input("Enter your password:", type="password")

    if st.button("Login"):
        if username in df["username"].values and password == "user@123":
            st.session_state.logged_in = True
            user_data = df[df["username"] == username]
            st.session_state.user_id = user_data["user_id"].values[0]

            model = user_data["model"].values[0]
            location = user_data["location"].values[0]
            
            st.write(f"Welcome, {username}! Your assigned model is '{model}'.")
            if st.button("Confirm Model"):
                st.session_state.model_confirmed = True
                pdf_content = process_pdf(location)
                st.session_state.retriever = create_retriever(pdf_content)
                chatbot_interface()
            else:
                st.error("Please confirm your model.")
        else:
            st.error("Invalid username or password. Please try again.")

# Chatbot interface
def chatbot_interface():
    st.write("**You can now ask questions about the manual!**")
    query = st.text_input("Ask a question:")

    if query and st.session_state.retriever:
        hf_pipeline = pipeline("text-generation", model="gpt2")
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.retriever,
        )
        response = qa_chain.run(query)
        st.write(f"**Answer:** {response}")

# Main logic
if st.session_state.logged_in:
    chatbot_interface()
else:
    login_and_model_selection()
