import pandas as pd
import streamlit as st

df = pd.read_csv("c:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\user_data.csv")

# Load manual data (example structure: company, model, location)
manual_data = pd.read_csv("C:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\laptop_models.csv")
manual_data.columns = ["company", "model", "location"]

# Streamlit interface
st.title("CHATBOT")
st.sidebar.title("Welcome to the Galaxy Office Automation Pvt Ltd")
st.sidebar.image(
    "c:\\Users\\Heeta Parmar\\Downloads\\Galaxy.png",  # Replace with the path to your image file
    caption="User Dashboard",
   
    use_container_width=True 
)

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

# Function to read PDF and display content
def read_pdf(file_path):
    try:
        pdf_document = fitz.open(file_path)
        text = ""
        for page in pdf_document:  # Iterate through pages
            text += page.get_text()  # Extract text from each page
        pdf_document.close()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

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

# Display model and manual location
def display_manual_info():
    user_id = st.session_state.user_id
    user_data = df[df["user_id"] == user_id]
    
    if not user_data.empty:
        model = user_data["model"].values[0]
        location = user_data["location"].values[0]

        if not st.session_state.model_confirmed:
            st.write(f"**Is your model '{model}'?**")
            
            if st.button("Yes"):
                st.session_state.model_confirmed = True
                st.write(f"**Manual Location:** {location}")
                pdf_text = read_pdf(location)  # Read and display the PDF
                st.text_area("Manual Content:", pdf_text, height=300)
            if st.button("No"):
                st.error("Model mismatch. Please select your company and model.")
                ask_for_company_and_model()
        else:
            st.write(f"**Your Model:** {model}")
            st.write(f"**Manual Location:** {location}")
            pdf_text = read_pdf(location)  # Read and display the PDF
            st.text_area("Manual Content:", pdf_text, height=300)
    else:
        st.error("No manual found for your user ID.")

# Ask for company and model
def ask_for_company_and_model():
    company = st.selectbox("Which company?", manual_data["company"].unique())
    st.session_state.company = company
    
    models = manual_data[manual_data["company"] == company]["model"].values
    model = st.selectbox("Select your model:", models)
    st.session_state.model = model

    location = manual_data[(manual_data["company"] == company) & (manual_data["model"] == model)]["location"].values[0]
    st.write(f"**Manual Location:** {location}")
    pdf_text = read_pdf(location)  # Read and display the PDF
    st.text_area("Manual Content:", pdf_text, height=300)

# Main application logic
if st.session_state.logged_in:
    st.header("Welcome to the User Dashboard")
    display_manual_info()
else:
    login_page()
#-----------------------------------------------------------
# import pandas as pd
# import streamlit as st

# # Load the CSV data
# df = pd.read_csv("c:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\user_data.csv")

# # Load manual data (example structure: company, model, location)
# manual_data = pd.read_csv("C:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\laptop_models.csv")
# manual_data.columns = ["company", "model", "location"]

# # Streamlit interface
# st.title("CHATBOT")
# st.sidebar.title("Welcome to the Galaxy Office Automation Pvt Ltd")
# st.sidebar.image(
#     "c:\\Users\\Heeta Parmar\\Downloads\\Galaxy.png",  # Replace with the path to your image file
#     caption="User Dashboard",
   
#     use_container_width=True 
# )

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
# if "manual_location" not in st.session_state:
#     st.session_state.manual_location = None

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

# # Ask for company and model
# def ask_for_company_and_model():
#     # Select the company
#     company = st.selectbox("Which company?", manual_data["company"].unique(), key="company_select")
#     st.session_state.company = company

#     # Filter available models based on the selected company
#     models = manual_data[manual_data["company"] == company]["model"].values
#     model = st.selectbox("Select your model:", models, key="model_select")
#     st.session_state.model = model

#     # Get the manual location for the selected company and model
#     location = manual_data[(manual_data["company"] == company) & (manual_data["model"] == model)]["location"].values[0]
#     st.session_state.manual_location = location

#     # Ask user if this is the correct model
#     st.write(f"Is your model: **{model}**?")
#     confirm_model = st.radio("Please confirm if this is the correct model:", ["Yes", "No"])

#     if confirm_model == "Yes":
#         st.session_state.model_confirmed = True
#         st.write(f"**Manual Location:** {location}")
#         st.download_button("Download Manual", location, file_name=model.replace(" ", "_") + ".pdf")
#     elif confirm_model == "No":
#         st.session_state.model_confirmed = False
#         st.error("The model is incorrect. Please select the correct company and model.")
#         # Optionally, you can reset the company and model selections here
#         st.session_state.company = None
#         st.session_state.model = None
#         st.session_state.manual_location = None

# # Main application logic
# if st.session_state.logged_in:
#     st.header("Welcome to the User Dashboard")
#     ask_for_company_and_model()
# else:
#     login_page()






