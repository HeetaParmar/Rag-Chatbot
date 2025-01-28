# # # #password 
# # # import streamlit as st
# # # # def password():
# # # #     password = input("Enter your password: ")
# # # #     if password == "password":
# # # #         print("Access granted")
# # # #     else:
# # # #         print("Access denied")
# # # # password()
# # # # # Extract text from a PDF
# # # # def extract_text_from_pdf(pdf_path):
# # # #     text = ""
# # # #     with open(pdf_path, "rb") as file:
# # # #         reader = PyPDF2.PdfReader(file)
# # # #         for page in reader.pages:
# # # #             text += page.extract_text()

# # # #     return text
# # import pandas as pd
# # import streamlit as st

# # # Load the CSV data
# # df = pd.read_csv("c:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\user_data.csv")

# # # Streamlit interface
# # st.title("CHATBOT")

# # # Initialize session state
# # if "logged_in" not in st.session_state:
# #     st.session_state.logged_in = False
# # if "user_id" not in st.session_state:
# #     st.session_state.user_id = None
# # if "model_confirmed" not in st.session_state:
# #     st.session_state.model_confirmed = False

# # # Login page
# # def login_page():
# #     st.title("Login")
# #     username = st.selectbox("Select your username:", ["user1", "user2", "user3", "user4", "user5", "user6"])
# #     password = st.text_input("Enter your password:", type="password")

# #     if st.button("Login"):
# #         user_id = int(username.replace("user", ""))  # Extract user_id from username (e.g., "user1" -> 1)
# #         if password == "user@123":
# #             st.session_state.logged_in = True
# #             st.session_state.user_id = user_id
# #             st.success(f"Login successful as {username}!")
# #         else:
# #             st.error("Invalid password. Please try again.")

# # # Display model and manual location
# # def display_manual_info():
# #     user_id = st.session_state.user_id
# #     user_data = df[df["user_id"] == user_id]
# #     if not user_data.empty:
# #         model = user_data["model"].values[0]
# #         location = user_data["location"].values[0]

# #         # Ask for user confirmation
# #         if not st.session_state.model_confirmed:
# #             st.write(f"**Is your model '{model}'?**")
# #             if st.button("Yes"):
# #                 st.session_state.model_confirmed = True
# #                 st.write(f"**Manual Location:** {location}")
# #                 st.download_button("Download Manual", location, file_name=model.replace(" ", "_") + ".pdf")
# #             elif st.button("No"):
# #                 st.error("Model mismatch. Please contact support.")
# #         else:
# #             st.write(f"**Your Model:** {model}")
# #             st.write(f"**Manual Location:** {location}")
# #             st.download_button("Download Manual", location, file_name=model.replace(" ", "_") + ".pdf")
# #     else:
# #         st.error("No manual found for your user ID.")

# # # Main application logic
# # if st.session_state.logged_in:
# #     st.header("Welcome to the User Dashboard")
# #     display_manual_info()
# # else:
# #     login_page()


# import pandas as pd
# import streamlit as st

# # Load the CSV data
# df = pd.read_csv("c:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\user_data.csv")

# # Streamlit interface
# st.title("CHATBOT")
# st.sidebar.title("Welcome to the User Dashboard")

# # Custom Sidebar Styling
# st.markdown("""
#     <style>
#         .css-1d391kg {
#             background-color: #f0f0f0;  /* Sidebar Background Color */
#         }
#         .css-2trq9p {
#             background-color: #4CAF50; /* Sidebar Title Color */
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
# if "user_id" not in st.session_state:
#     st.session_state.user_id = None
# if "model_confirmed" not in st.session_state:
#     st.session_state.model_confirmed = False
# if "company" not in st.session_state:
#     st.session_state.company = None
# if "model" not in st.session_state:
#     st.session_state.model = None

# # Manual PDF location data
# manual_data = pd.read_csv("C:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\laptop_models.csv")

# # Login page
# def login_page():
#     st.title("Login")
#     username = st.selectbox("Select your username:", ["user1", "user2", "user3", "user4", "user5", "user6"])
#     password = st.text_input("Enter your password:", type="password")
    
#     if st.button("Login"):
#         user_id = int(username.replace("user", ""))  # Extract user_id from username (e.g., "user1" -> 1)
#         if password == "user@123":
#             st.session_state.logged_in = True
#             st.session_state.user_id = user_id
#             st.success(f"Login successful as {username}!")
#         else:
#             st.error("Invalid password. Please try again.")

# # Display model and manual location
# def display_manual_info():
#     user_id = st.session_state.user_id
#     user_data = df[df["user_id"] == user_id]
    
#     if not user_data.empty:
#         model = user_data["model"].values[0]
#         location = user_data["location"].values[0]

#         # Ask for user confirmation about the model
#         if not st.session_state.model_confirmed:
#             st.write(f"**Is your model '{model}'?**")
            
#             # Custom colored buttons for confirmation
#             col1, col2 = st.columns([1, 1])
#             with col1:
#                 if st.button("Yes", key="yes_button", help="Confirm your model", use_container_width=True):
#                     st.session_state.model_confirmed = True
#                     st.write(f"**Manual Location:** {location}")
#                     st.download_button("Download Manual", location, file_name=model.replace(" ", "_") + ".pdf")
#             with col2:
#                 if st.button("No", key="no_button", help="Incorrect model", use_container_width=True):
#                     st.session_state.model_confirmed = False
#                     st.session_state.company = None
#                     st.session_state.model = None
#                     st.session_state.model_confirmed = False
#                     st.error("Model mismatch. Please select your company and model.")
#                     ask_for_company_and_model()

#         else:
#             st.write(f"**Your Model:** {model}")
#             st.write(f"**Manual Location:** {location}")
#             st.download_button("Download Manual", location, file_name=model.replace(" ", "_") + ".pdf")
#     else:
#         st.error("No manual found for your user ID.")

# def ask_for_company_and_model():
#     # Ask for company selection
#     company = st.selectbox("Which company?", ["dell", "lenovo", "microsoft"])
#     st.session_state.company = company
    
#     # Based on company, list available models
#     models = list(manual_data[company].keys())
#     model = st.selectbox("Select your model:", models)
#     st.session_state.model = model

#     # Show the manual location for the selected company and model
#     location = manual_data[company][model]
#     st.write(f"**Manual Location:** {location}")
#     st.download_button("Download Manual", location, file_name=model.replace(" ", "_") + ".pdf")

# # Main application logic
# if st.session_state.logged_in:
#     st.header("Welcome to the User Dashboard")
#     display_manual_info()
# else:
#     login_page()

# pyqyt 5

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import openai
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import os

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")



client = openai.OpenAI(
    api_key="d72661ce-be9a-482e-a14e-ab2302105b4f",
    base_url="https://api.sambanova.ai/v1",
)

response = client.chat.completions.create(
    model='Meta-Llama-3.1-8B-Instruct',
    messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"}],
    temperature =  0.1,
    top_p = 0.1
)

print(response.choices[0].message.content)
      

# Load the CSV data
csv_file_path = pd.read_csv("c:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\user_data.csv")
import pandas as pd

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Convert to JSON
json_file_path = "your_json_file.json"  # Replace with your desired JSON file path
df.to_json(json_file_path, orient="records", indent=4)

print(f"CSV has been successfully converted to JSON and saved at {json_file_path}.")


# Streamlit interface
st.title("CHATBOT")
st.sidebar.title("Welcome to the Galaxy Office Automation Pvt Ltd")
st.sidebar.image(
    "c:\\Users\\Heeta Parmar\\Downloads\\Galaxy.png",  # Replace with the path to your image file
    caption="User Dashboard",
    use_container_width=True 
)

# Custom Sidebar Styling
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #f0f0f0;  /* Sidebar Background Color */
        }
        .css-2trq9p {
            background-color: #4CAF50; /* Sidebar Title Color */
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "model_confirmed" not in st.session_state:
    st.session_state.model_confirmed = False
if "company" not in st.session_state:
    st.session_state.company = None
if "model" not in st.session_state:
    st.session_state.model = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Manual PDF location data
manual_data = pd.read_csv("C:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\laptop_models.csv")

# Login page
# Login page
def login_page():
    st.title("Login")
    username = st.selectbox("Select your username:", ["user1", "user2", "user3", "user4", "user5", "user6"])
    password = st.text_input("Enter your password:", type="password")

    if st.button("Login"):
        user_id = int(username.replace("user", ""))  # Extract user_id from username (e.g., "user1" -> 1)
        if password == "user@123":
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.success(f"Login successful as {username}!")
        else:
            st.error("Invalid password. Please try again.")


# Read and process PDF content
def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Create a retriever for chatbot functionality
def create_retriever(pdf_content):
    embeddings = HuggingFaceEmbeddings(model_name="facebook/dpr-question_encoder-multiset-base")
    vectorstore = FAISS.from_texts([pdf_content], embeddings)
    return vectorstore.as_retriever()

def get_chunks(pdf_path):
    pdf_document = fitz.open(pdf_path)
    toc = pdf_document.get_toc()
    if toc:
        sections_list = []
        for i in range(len(toc)):
            title = toc[i][1]
            start_page = toc[i][2]
            end_page = (
                toc[i + 1][2] - 1 if i + 1 < len(toc) else pdf_document.page_count
            )  # Get end page based on next TOC entry

            # Extract text for the section
            section_text = extract_section(pdf_document, start_page, end_page, title)

            if section_text:
                sections_list.append(section_text)
    else:
        sections_list = []

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)  # Load the page
            page_text = page.get_text("text")  # Extract text from the page
            sections_list.append(page_text)

    chunks = sections_list
    return chunks
    
    
    # Chatbot functionality
def chatbot_interface():
    st.write("**You can now ask questions about the manual!**")
    query = st.text_input("Ask a question:")

    if query and st.session_state.retriever:
        # Set up Hugging Face pipeline for LLM
        hf_pipeline = pipeline("text-generation", model="gpt2")
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=st.session_state.retriever
        )
        response = qa_chain.run(query)
        st.write(f"**Answer:** {response}")

# Display model and manual location
def display_manual_info():
    user_id = st.session_state.user_id
    user_data = df[df["user_id"] == user_id]

    if not user_data.empty:
        model = user_data["model"].values[0]
        location = user_data["location"].values[0]

        # Ask for user confirmation about the model
        if not st.session_state.model_confirmed:
            st.write(f"**Is your model '{model}'?**")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Yes", key="yes_button", help="Confirm your model", use_container_width=True):
                    st.session_state.model_confirmed = True
                    st.write(f"**Manual Location:** {location}")

                    # Process the PDF and initialize retriever
                    pdf_content = process_pdf(location)
                    print(pdf_content[:100])  # Debugging line to check the content
                    st.session_state.retriever = create_retriever(pdf_content)

                    # Display chatbot interface
                    chatbot_interface()
            with col2:
                if st.button("No", key="no_button", help="Incorrect model", use_container_width=True):
                    st.session_state.model_confirmed = False
                    st.session_state.company = None
                    st.session_state.model = None
                    st.session_state.model_confirmed = False
                    st.error("Model mismatch. Please select your company and model.")
                    ask_for_company_and_model()
        else:
            st.write(f"**Your Model:** {model}")
            st.write(f"**Manual Location:** {location}")

            # Process the PDF and initialize retriever
            pdf_content = process_pdf(location)
            st.session_state.retriever = create_retriever(pdf_content)

            # Display chatbot interface
            chatbot_interface()
    else:
        st.error("No manual found for your user ID.")

def ask_for_company_and_model():
    company = st.selectbox("Which company?", ["dell", "lenovo", "microsoft"])
    st.session_state.company = company

    models = list(manual_data[company].keys())
    model = st.selectbox("Select your model:", models)
    st.session_state.model = model

    location = manual_data[company][model]
    st.write(f"**Manual Location:** {location}")
    st.download_button("Download Manual", location, file_name=model.replace(" ", "_") + ".pdf")

# Main application logic
if st.session_state.logged_in:
    st.header("Welcome to the User Dashboard")
    display_manual_info()
else:
    login_page()



