from flask import Flask, request, render_template_string
import json
import os
import fitz  # PyMuPDF for extracting text from PDFs
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import numpy as np
import os
import openai
# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("feature-extraction", model="facebook/dpr-question_encoder-multiset-base")
# Load model directly
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
model = AutoModel.from_pretrained("facebook/dpr-question_encoder-multiset-base")


client = openai.OpenAI(
    api_key=os.environ.get("5c19253d-8fa6-4517-99fe-87b9dd0590b6"),
    base_url="https://api.sambanova.ai/v1",
)

response = client.chat.completions.create(
    model='Meta-Llama-3.1-8B-Instruct',
    messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"}],
    temperature =  0.1,
    top_p = 0.1
)

print(response.choices[0].message.content)
      

# Load user data from JSON files
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

user_data = load_json("user_data.json")
manuals_data = load_json("manual_data.json")

app = Flask(__name__)

# Ensure the static directory exists
if not os.path.exists("static"):
    os.makedirs("static/manuals")

# Initialize the DPR model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base").to(device)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

# Helper function to generate embeddings
def encode_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        inputs = context_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
        embedding = context_encoder(**inputs).pooler_output.detach().cpu().numpy()
        embeddings.append(embedding)
    return embeddings

# Function to save embeddings to disk
def save_embeddings(embeddings, username):
    embeddings_dir = "static/embeddings"
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    # Save each embedding in a separate file
    for i, embedding in enumerate(embeddings):
        embedding_file = os.path.join(embeddings_dir, f"{username}_embedding_{i + 1}.npy")
        np.save(embedding_file, embedding)  # Save as .npy file

# Login Page Template
login_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Welcome to Galaxy Office Automation</title>
</head>
<body>
    <center>
    <h1>Welcome to Galaxy Office Automation</h1>
    <form method="POST" action="/login">
        <h1><label for="username_id">Username:</label>
        <input type="text" id="username" name="username" required><br></h1>
        <h1><label for="password">Password:</label>
        <input type="password" id="password" name="password" required><br></h1>
        <h1><button type="submit"><h3>Login</h3></button></h1>
    </center>
    </form>
</body>
</html>
"""

# Model Selection Template
model_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Select Model</title>
</head>
<body>
    <h1>Welcome User {{ username }}</h1>
    <h2><b><p>Are you using the model: {{ model }}?</h2></b></p>
    <form method="POST" action="/validate_model">
        <input type="hidden" name="username" value="{{ username }}">
        <input type="hidden" name="model" value="{{ model }}">
        <button type="submit" name="response" value="yes">Yes</button>
        <button type="submit" name="response" value="no">No</button>
    </form>
</body>
</html>
"""

# Company Selection Template
company_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Select Company</title>
</head>
<body>
    <h1>Select a Company</h1>
    <form method="POST" action="/select_model">
        <input type="hidden" name="username" value="{{ username }}">
        <label for="company">Company:</label>
        <select name="company" id="company" required>
            {% for company in companies %} 
                <option value="{{ company }}">{{ company }}</option>
            {% endfor %}
        </select><br>
        <h1><button type="submit">Next</button></h1>
    </form>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(login_page)

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    user = next((user for user in user_data if str(user["user_id"]) == username), None)

    if user:
        if "password" in user and user["password"] == password:
            return render_template_string(model_page, username=username, model=user["model"])
        elif "password" not in user:
            return render_template_string(model_page, username=username, model=user["model"])
        else:
            return "<h1>Invalid password</h1>", 401
    else:
        return "<h1>Invalid login ID</h1>", 401

@app.route("/validate_model", methods=["POST"])
def validate_model():
    username = request.form["username"]
    response = request.form["response"]
    model = request.form["model"]

    if response == "yes":
        user = next((user for user in user_data if str(user["user_id"]) == username), None)
        if user:
            pdf_path = user.get("location")
            if pdf_path and os.path.exists(pdf_path):
                pdf_text = extract_pdf_text(pdf_path)
                chunks = chunk_text(pdf_text)
                embeddings = encode_chunks(chunks)

                # Save the embeddings to disk
                save_embeddings(embeddings,username)

                return f"""
                    <h1>Welcome, user {username}!</h1>
                    <h2>You have selected the model: {model}.</h2>
                    <h3>Embeddings have been saved successfully.</h3>
                    <p>You can now use these embeddings for further tasks.</p>
                """
            else:
                return f"""
                    <h1>Error:</h1>
                    <p>PDF for the model '{model}' not found or does not exist at '{pdf_path}'.</p>
                """, 404
        else:
            return "<h1>Error: User data not found.</h1>", 404

    else:
        companies = sorted(set(manual["company"] for manual in manuals_data))
        return render_template_string(company_page, username=username, companies=companies)

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return "Error extracting text from PDF."

def chunk_text(text, chunk_size=512):
    """Split the text into chunks of a specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

if __name__ == "__main__":
    app.run(debug=True)



