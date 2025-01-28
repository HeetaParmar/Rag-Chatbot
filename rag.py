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
        {% for message in chat_history %}
            <p><strong>{{ message.role.capitalize() }}:</strong> {{ message.content }}</p>
        {% endfor %}
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

                # Save chunks to a file for persistence
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

@app.route("/ask_question", methods=["POST"])
def ask_question():
    global stored_embeddings, stored_chunks

    model = request.form["model"]
    question = request.form["question"]

    # Ensure embeddings are available
    if not stored_embeddings or not stored_chunks:
        return "<h1>Error: No embeddings available. Please select a document first.</h1>", 400

    # Generate embedding for the user query
    question_embedding = context_encoder.encode(question, convert_to_tensor=True, device=device).cpu().numpy()

    # Compute cosine similarity between query and stored embeddings
    similarities = [
        torch.nn.functional.cosine_similarity(
            torch.tensor(question_embedding).unsqueeze(0),  # Add batch dimension
            torch.tensor(chunk_embedding).unsqueeze(0),  # Add batch dimension
            dim=1
        ).item()
        for chunk_embedding in stored_embeddings
    ]

    # Handle empty similarities
    if not similarities:
        return "<h1>Error: Unable to calculate similarities. Check your embeddings or query.</h1>", 400

    # Find the most relevant chunk
    most_relevant_index = similarities.index(max(similarities))
    most_relevant_chunk = stored_chunks[most_relevant_index]

    # Initialize or retrieve the chat history from the session
    if "chat_history" not in session:
        session["chat_history"] = []

    # Add the retrieved context to the system role
    conversation_history = session["chat_history"]
    conversation_history.append({"role": "system", "content": most_relevant_chunk})
    conversation_history.append({"role": "user", "content": question})

    # Generate response using the language model
    response = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",
        messages=conversation_history,
        temperature=0.1,
        top_p=0.1,
    )

    answer = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})

    # Save the updated chat history to the session
    session["chat_history"] = conversation_history

    # Render the chatbot page with the updated chat history
    return render_template_string(chatbot_page, model=model, chat_history=conversation_history)

@app.route("/end_chat", methods=["POST"])
def end_chat():
    # Clear chat history from the session
    session.pop("chat_history", None)

    # Reset the stored embeddings and chunks (optional, based on your use case)
    global stored_embeddings, stored_chunks
    stored_embeddings = []
    stored_chunks = []

    # Return a message or redirect to the home page or login
    return "<h1>Chat ended. You can start a new session.</h1>", 200

def get_chunks(pdf_path):
    pdf_document = fitz.open(pdf_path)
    toc = pdf_document.get_toc()
    sections_list = []

    if toc:
        for i in range(len(toc)):
            title = toc[i][1]
            start_page = toc[i][2]
            end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else pdf_document.page_count
            section_text = extract_section(pdf_document, start_page, end_page, title)
            if section_text:
                sections_list.append(section_text)
    else:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text("text")
            sections_list.append(page_text)

    pdf_document.close()
    return sections_list

def extract_section(pdf_document, start_page, end_page, title):
    section_text = f"Section Title: {title}\n\n"
    for page_num in range(start_page - 1, end_page):
        page = pdf_document.load_page(page_num)
        section_text += page.get_text("text")
    return section_text

def convert_chunks_to_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = context_encoder.encode(chunk, convert_to_tensor=True, device=device)
        embeddings.append(embedding.cpu().numpy())
    return embeddings

# Save chunks to a file for persistence
def save_chunks_to_file(chunks, filename="chunks.json"):
    with open(filename, 'w') as f:
        json.dump(chunks, f)

# Load chunks from a file
def load_chunks_from_file(filename="chunks.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

if __name__ == "__main__":
    app.run(debug=True)
