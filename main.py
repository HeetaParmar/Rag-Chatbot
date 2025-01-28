import pandas as pd
import streamlit as st

# # Load the CSV data
df = pd.read_csv("c:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\user_data.csv")

# # Streamlit interface
st.title("CHATBOT")

# # Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "model_confirmed" not in st.session_state:
    st.session_state.model_confirmed = False

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

        # Ask for user confirmation
        if not st.session_state.model_confirmed:
            st.write(f"**Is your model '{model}'?**")
            if st.button("Yes"):
                st.session_state.model_confirmed = True
                st.write(f"**Manual Location:** {location}")
                st.download_button("Download Manual", location, file_name=model.replace(" ", "_") + ".pdf")
            elif st.button("No"):
                st.error("Model mismatch. Please contact support.")
        else:
            st.write(f"**Your Model:** {model}")
            st.write(f"**Manual Location:** {location}")
            st.download_button("Download Manual", location, file_name=model.replace(" ", "_") + ".pdf")
    else:
        st.error("No manual found for your user ID.")

# Main application logic
if st.session_state.logged_in:
    st.header("Welcome to the User Dashboard")
    display_manual_info()
else:
    login_page()
