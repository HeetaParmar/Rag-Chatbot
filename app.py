from flask import Flask, request, render_template_string, jsonify, session
import json
import os
import fitz  # PyMuPDF for extracting text from PDFs
from sentence_transformers import SentenceTransformer
import torch
import openai

# Initialize OpenAI client
client = openai.OpenAI(
    api_key="559480d3-5c94-47ed-b1e1-4d9a30fb51f9",
    base_url="https://api.sambanova.ai/v1",
)

# Load user data from JSON files
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

user_data = load_json("user_data.json")
manuals_data = load_json("manual_data.json")

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Replace with a secure key in production

# Initialize Sentence-Transformers model for embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2").to(device)

# Globals for storing embeddings and chunks
stored_embeddings = []
stored_chunks = []

# HTML Templates
login_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Login to Galaxy Office Automation</title>
</head>
<body>
    <center>
    <h1>Welcome to Galaxy Office Automation</h1>
    <form method="POST" action="/login">
        <label for="username_id">Username:</label>
        <input type="text" id="username" name="username" required><br><br>
        <label for="password">Password:</label>
        <input type="password" id="password" name="password" required><br><br>
        <button type="submit">Login</button>
    </form>
    </center>
</body>
</html>
"""

model_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Select Model</title>
</head>
<body>
    <h1>Welcome User {{ username }}</h1>
    <h2>Are you using the model: {{ model }}?</h2>
    <form method="POST" action="/validate_model">
        <input type="hidden" name="username" value="{{ username }}">
        <input type="hidden" name="model" value="{{ model }}">
        <button type="submit" name="response" value="yes">Yes</button>
        <button type="submit" name="response" value="no">No</button>
    </form>
</body>
</html>
"""

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

model_list_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Select Model</title>
</head>
<body>
    <h1>Select a Model</h1>
    <form method="POST" action="/finalize">
        <input type="hidden" name="username" value="{{ username }}">
        <label for="model">Model:</label>
        <select name="model" id="model" required>
            {% for model in models %}
                <option value="{{ model }}">{{ model }}</option>
            {% endfor %}
        </select><br>
        <h1><button type="submit">Submit</button></h1>
    </form>
</body>
</html>
"""

chatbot_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Chat with Model</title>
</head>
<body>
    <h1>Chat with the Model: {{ model }}</h1>
    <div style="border: 1px solid #ccc; padding: 10px; margin-bottom: 20px;">
        <h3>Chat History:</h3>
        <ul>
            {% for message in chat_history %}
                <li><strong>{{ message.role.capitalize() }}:</strong> {{ message.content }}</li>
            {% endfor %}
        </ul>
    </div>
    <form method="POST" action="/ask_question">
        <input type="hidden" name="model" value="{{ model }}">
        <label for="question">Ask a Question:</label>
        <input type="text" id="question" name="question" required><br><br>
        <button type="submit">Ask</button>
    </form>
    <form method="POST" action="/end_chat">
        <input type="hidden" name="model" value="{{ model }}">
        <button type="submit">End Chat</button>
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
                chunks = get_chunks(pdf_path)
                embeddings = convert_chunks_to_embeddings(chunks)

                global stored_embeddings, stored_chunks
                stored_embeddings = embeddings
                stored_chunks = chunks

                save_chunks_to_file(stored_chunks)

                return render_template_string(chatbot_page, model=model, chat_history=[])
            else:
                return f"<h1>Error:</h1><p>PDF for the model {model} not found or does not exist at '{pdf_path}'.</p>", 404
    else:
        companies = sorted(set(manual["company"] for manual in manuals_data))
        return render_template_string(company_page, username=username, companies=companies)

@app.route("/select_model", methods=["POST"])
def select_model():
    username = request.form["username"]
    company = request.form["company"]

    models = [manual["model"] for manual in manuals_data if manual["company"] == company]
    return render_template_string(model_list_page, username=username, models=models)

@app.route("/finalize", methods=["POST"])
def finalize():
    username = request.form["username"]
    model = request.form["model"]

    manual = next((manual for manual in manuals_data if manual["model"] == model), None)
    if manual:
        pdf_path = manual.get("location")
        if pdf_path and os.path.exists(pdf_path):
            chunks = get_chunks(pdf_path)
            embeddings = convert_chunks_to_embeddings(chunks)

            global stored_embeddings, stored_chunks
            stored_embeddings = embeddings
            stored_chunks = chunks

            save_chunks_to_file(stored_chunks)

            return render_template_string(chatbot_page, model=model, chat_history=[])
        else:
            return f"<h1>Error:</h1><p>PDF for the model {model} not found or does not exist at '{pdf_path}'.</p>", 404
    else:
        return f"<h1>Error:</h1><p>Model {model} not found in the database.</p>", 404

@app.route("/ask_question", methods=["POST"])
def ask_question():
    global stored_embeddings, stored_chunks

    model = request.form["model"]
    question = request.form["question"]

    if not stored_embeddings or not stored_chunks:
        return "<h1>Error: No embeddings available. Please select a document first.</h1>", 400

    question_embedding = context_encoder.encode(question, convert_to_tensor=True, device=device).cpu().numpy()

    similarities = [
        torch.nn.functional.cosine_similarity(
            torch.tensor(question_embedding).unsqueeze(0),
            torch.tensor(chunk_embedding).unsqueeze(0),
            dim=1
        ).item()
        for chunk_embedding in stored_embeddings
    ]

    if not similarities:
        return "<h1>Error: Unable to calculate similarities. Check your embeddings or query.</h1>", 400

    most_relevant_index = similarities.index(max(similarities))
    most_relevant_chunk = stored_chunks[most_relevant_index]

    if "chat_history" not in session:
        session["chat_history"] = []

    conversation_history = session["chat_history"]
    conversation_history.append({"role": "system", "content": most_relevant_chunk})
    conversation_history.append({"role": "user", "content": question})

    response = client.chat_conversations(
        request={
            "prompt": conversation_history,
        }
    )

