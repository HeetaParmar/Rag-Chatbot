from flask import Flask, request, render_template_string, session, send_file, url_for
import json
import os
import fitz  # PyMuPDF for extracting text from PDFs
from sentence_transformers import SentenceTransformer
import torch
import openai
from gtts import gTTS  # Import gTTS for text-to-speech
import uuid
from werkzeug.utils import secure_filename

# Initialize OpenAI client
client = openai.OpenAI(
    api_key="a34728d9-5b9d-4e4b-b8d8-c1c991eb9c70",
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
stored_metadata=[]
# HTML Templates
login_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Login to Galaxy Office Automation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(120deg, #ff0000, #ffffff, #0000ff);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
</head>
<body>
    <div class="container text-center">
        <h1 class="my-5">Welcome to Galaxy Office Automation</h1>
        <form method="POST" action="/login" class="border p-4 bg-white shadow-sm rounded">
            <div class="mb-3">
                <label for="username" class="form-label">Username:</label>
                <input type="text" id="username" name="username" class="form-control" required>
            </div>
            <div class="mb-3">
                <label for="password" class="form-label">Password:</label>
                <input type="password" id="password" name="password" class="form-control" required>
            </div>
            <button type="submit" class="btn btn-primary w-100">Login</button>
        </form>
    </div>
</body>
</html>
"""
model_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Select Model</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(120deg, #ff0000, #ffffff, #0000ff);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
</head>
<body class="bg-light">
    <div class="container text-center">
        <h1 class="my-5">Welcome, {{ username }}</h1>
        <p>Are you using the model: <strong>{{ model }}</strong>?</p>
        <form method="POST" action="/validate_model" class="d-inline-block">
            <input type="hidden" name="username" value="{{ username }}">
            <input type="hidden" name="model" value="{{ model }}">
            <button type="submit" name="response" value="yes" class="btn btn-success me-2">Yes</button>
            <button type="submit" name="response" value="no" class="btn btn-danger">No</button>
        </form>
        <div class="mt-4">
            <a href="{{ back_url }}" class="btn btn-secondary">Back</a>
        </div>
    </div>
</body>
</html>
"""

company_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Select Company</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(120deg, #ff0000, #ffffff, #0000ff);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
</head>
<body class="bg-light">
    <div class="container text-center">
        <h1 class="my-5">Select a Company</h1>
        <form method="POST" action="/select_model" class="border p-4 bg-white shadow-sm rounded">
            <input type="hidden" name="username" value="{{ username }}">
            <div class="mb-3">
                <label for="company" class="form-label">Company:</label>
                <select name="company" id="company" class="form-select" required>
                    {% for company in companies %}
                        <option value="{{ company }}">{{ company }}</option>
                    {% endfor %}
                </select>
            </div>
            <button type="submit" class="btn btn-primary w-100">Next</button>
        </form>
        <div class="mt-4">
            <a href="{{ back_url }}" class="btn btn-secondary">Back</a>
        </div>
    </div>
</body>
</html>
"""

model_list_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Select Model</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(120deg, #ff0000, #ffffff, #0000ff);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
</head>
<body class="bg-light">
    <div class="container text-center">
        <h1 class="my-5">Select a Model</h1>
        <form method="POST" action="/select_model" class="border p-4 bg-white shadow-sm rounded">
            <input type="hidden" name="username" value="{{ username }}">
            <input type="hidden" name="company" value="{{ company }}">
            <div class="mb-3">
                <label for="model" class="form-label">Model:</label>
                <select name="model" id="model" class="form-select" required>
                    {% for model in models %}
                        <option value="{{ model }}">{{ model }}</option>
                    {% endfor %}
                </select>
            </div>
            <button type="submit" class="btn btn-primary w-100">Submit</button>
        </form>
        <div class="mt-4">
            <a href="{{ back_url }}" class="btn btn-secondary">Back</a>
        </div>
    </div>
</body>
</html>
"""


# HTML Templates
chatbot_page = """
<!doctype html>
<html lang="en">
<head>
    <title>Chat with Model</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script>
        // Web Speech API for Speech-to-Text
        function startRecognition() {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'en-US';
            recognition.interimResults = false;

            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                document.getElementById('question').value = transcript;
            };

            recognition.onerror = function(event) {
                alert('Speech recognition error: ' + event.error);
            };

            recognition.start();
        }
    </script>
    <style>
        body {
            background: linear-gradient(120deg, #ff0000, #ffffff, #0000ff);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chat-message {
            white-space: pre-wrap;
        }
    </style>
</head>
<body class="bg-light">
    <div class="container">
        <h1 class="text-center my-5">Chat with Model</h1>
        
        <!-- Chat History -->
        <div class="border p-4 bg-white shadow-sm rounded mb-4">
            <h3>Chat History:</h3>
            <ul class="list-group">
                {% for message in chat_history %}
                    <li class="list-group-item">
                        <strong>{{ message.role.capitalize() }}:</strong> 
                        <p class="chat-message">{{ message.content|safe }}</p>
                        {% if message.audio_file %}
                            <br>
                            <audio controls class="mt-2">
                                <source src="{{ message.audio_file }}" type="audio/mpeg">
                                Your browser does not support the audio element.
                            </audio>
                        {% endif %}
                    </li>
                {% endfor %}
            </ul>
        </div>

        <!-- Form to Ask a New Question -->
        <form method="POST" action="/ask_question" class="d-flex flex-column align-items-center mb-3" enctype="multipart/form-data">
            <input type="hidden" name="model" value="{{ model }}">
            <div class="mb-3 w-100">
                <label for="question" class="form-label">Ask a Question:</label>
                <div class="input-group">
                    <input type="text" id="question" name="question" class="form-control" placeholder="Type or speak your question...">
                    <button type="button" class="btn btn-secondary" onclick="startRecognition()">🎤 Speak</button>
                </div>
            </div>
            <button type="submit" class="btn btn-primary w-100">Submit</button>
        </form>

        <!-- Confirmation Buttons (Yes/No) -->
        {% if chat_history and chat_history[-1].role == 'assistant' and 'Have you completed this step?' in chat_history[-1].content %}
            <div class="d-flex justify-content-center gap-2 mb-3">
                <form method="POST" action="/ask_question" class="d-inline">
                    <input type="hidden" name="model" value="{{ model }}">
                    <input type="hidden" name="user_response" value="yes">
                    <button type="submit" class="btn btn-success">Yes</button>
                </form>
                <form method="POST" action="/ask_question" class="d-inline">
                    <input type="hidden" name="model" value="{{ model }}">
                    <input type="hidden" name="user_response" value="no">
                    <button type="submit" class="btn btn-danger">No</button>
                </form>
            </div>
        {% endif %}

        <!-- Clear and End Chat Buttons -->
        <div class="d-flex justify-content-between">
            <a href="/" class="btn btn-secondary">Back</a>
            <div>
                <form method="POST" action="/clear_chat" class="d-inline">
                    <button type="submit" class="btn btn-warning me-2">Clear Chat</button>
                </form>
                <form method="POST" action="/end_chat" class="d-inline">
                    <button type="submit" class="btn btn-danger">End Chat</button>
                </form>
            </div>
        </div>
    </div>
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
            return render_template_string(model_page, username=username, model=user["model"], back_url="/")
        elif "password" not in user:
            return render_template_string(model_page, username=username, model=user["model"], back_url="/")
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
                chunks, metadata = get_chunks(pdf_path)  # Fetch chunks and metadata
                embeddings = convert_chunks_to_embeddings(chunks)

                global stored_embeddings, stored_chunks, stored_metadata
                stored_embeddings = embeddings
                stored_chunks = chunks
                stored_metadata = metadata  # Store metadata globally

                return render_template_string(chatbot_page, model=model, chat_history=[], back_url="/validate_model")
            else:
                return f"<h1>Error:</h1><p>PDF for the model {model} not found or does not exist at '{pdf_path}'.</p>", 404
    else:
        companies = sorted(set(manual["company"] for manual in manuals_data))
        return render_template_string(company_page, username=username, companies=companies, back_url="/validate_model")

@app.route("/select_model", methods=["POST"])
def select_model():
    username = request.form["username"]
    company = request.form.get("company")
    model = request.form.get("model", None)

    if not model:
        models = [manual["model"] for manual in manuals_data if manual["company"] == company]
        return render_template_string(model_list_page, username=username, company=company, models=models, back_url="/company_page")

    selected_manual = next((manual for manual in manuals_data if manual["model"] == model), None)
    if not selected_manual:
        return f"<h1>Error: Model '{model}' not found.</h1>", 404

    pdf_path = selected_manual.get("location")
    if not pdf_path or not os.path.exists(pdf_path):
        return f"<h1>Error: PDF for the model '{model}' not found at '{pdf_path}'.</h1>", 404

    chunks, metadata = get_chunks(pdf_path)
    embeddings = convert_chunks_to_embeddings(chunks)

    global stored_embeddings, stored_chunks, stored_metadata
    stored_embeddings = embeddings
    stored_chunks = chunks
    stored_metadata = metadata

    return render_template_string(chatbot_page, model=model, chat_history=[], back_url="/company_page")

@app.route("/ask_question", methods=["POST"])
def handle_decision_tree_logic(user_query, user_response, chat_history, decision_trees):
    # Check if the user query matches a specific problem
    matched_problem = None
    for problem, steps in decision_trees.items():
        if problem in user_query:
            matched_problem = problem
            break

    if matched_problem:
        current_problem = session.get("current_problem")
        current_step = session.get("current_step", 0)

        if current_problem is None:
            session["current_problem"] = matched_problem
            current_problem = matched_problem

        steps = decision_trees[current_problem]

        # Handle user response to the current step
        if user_response == "yes":
            if current_step + 1 < len(steps):
                session["current_step"] += 1
                chat_history.append({"role": "assistant", "content": steps[current_step + 1]})
            else:
                chat_history.append({
                    "role": "assistant",
                    "content": "All steps completed. If the issue persists, contact support."
                })
        elif user_response == "no":
            chat_history.append({
                "role": "assistant",
                "content": f"Please perform this step: {steps[current_step]}"
            })
    return chat_history

def handle_pdf_search(user_query, stored_embeddings, stored_chunks, chat_history):
    query_embedding = context_encoder.encode(user_query, convert_to_tensor=True, device=device).cpu().numpy()
    similarities = [
        torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding),
            torch.tensor(chunk_embedding),
            dim=0
        ).item()
        for chunk_embedding in stored_embeddings
    ]

    if similarities:
        most_relevant_index = similarities.index(max(similarities))
        most_relevant_chunk = stored_chunks[most_relevant_index]
        chat_history.append({"role": "assistant", "content": most_relevant_chunk})
    else:
        chat_history.append({
            "role": "assistant",
            "content": "No relevant information found in the document. Try another query."
        })
    return chat_history

@app.route("/ask_question", methods=["POST"])

def ask_question():
    global stored_embeddings, stored_chunks
    model = request.form["model"]
    user_query = request.form.get("question", "").lower()
    user_response = request.form.get("user_response")  # Yes/No responses

    if "chat_history" not in session:
        session["chat_history"] = [{"role": "system", "content": "How may I help you?"}]
        session["current_problem"] = None
        session["current_step"] = 0

    chat_history = session["chat_history"]

    # Decision Tree Logic
    decision_trees = {
        
        "display not working": [
            "Step 1: Check if the display is powered on. Ensure the power cable is connected properly.",
            "Step 2: Verify that the display is receiving input from the correct source (e.g., HDMI, VGA).",
            "Step 3: Check for any loose cables between the display and the computer.",
            "Step 4: Restart the display and the connected computer.",
            "Step 5: Test the display with another device to rule out hardware issues.",
            "Step 6: Contact technical support for further assistance.",
        ],
        "printer not working": [
            "Step 1: Ensure the printer is powered on and connected to the network.",
            "Step 2: Check for any paper jams in the printer.",
            "Step 3: Verify that the printer has sufficient ink or toner.",
            "Step 4: Restart the printer and your computer.",
            "Step 5: Reinstall or update the printer drivers.",
            "Step 6: Contact technical support for further assistance.",
        ],
        "internet not working": [
            "Step 1: Check if your device is connected to the Wi-Fi network.",
            "Step 2: Restart your router and modem.",
            "Step 3: Run a speed test to check for connectivity issues.",
            "Step 4: Verify if other devices on the network are facing the same issue.",
            "Step 5: Contact your Internet Service Provider for assistance.",
        ],
    }
    chat_history = handle_decision_tree_logic(user_query, user_response, chat_history, decision_trees)

    # PDF Search Logic
    if not user_response:
        chat_history = handle_pdf_search(user_query, stored_embeddings, stored_chunks, chat_history)

    session["chat_history"] = chat_history
    return render_template_string(chatbot_page, model=model, chat_history=chat_history, back_url="/")




    # Check if the user query matches any specific problem
    matched_problem = None
    for problem, steps in decision_trees.items():
        if problem in user_query:
            matched_problem = problem
            break

    if matched_problem:
        # Use the decision tree logic for specific problems
        if current_problem is None:
            session["current_problem"] = matched_problem
            current_problem = matched_problem

        steps = decision_trees[current_problem]

        # Handle user response to the current step
        if user_response:
            if user_response == "yes":
                if current_step + 1 < len(steps):
                    current_step += 1  # Move to the next step
                else:
                    chat_history.append({
                        "role": "assistant",
                        "content": "All steps have been completed. If the problem persists, please contact technical support."
                    })
                    session["chat_history"] = chat_history
                    return render_template_string(chatbot_page, model=model, chat_history=chat_history, back_url="/")
            elif user_response == "no":
                # Reiterate the current step
                chat_history.append({
                    "role": "assistant",
                    "content": f"Please perform this step: {steps[current_step]}"
                })
                session["chat_history"] = chat_history
                return render_template_string(chatbot_page, model=model, chat_history=chat_history, back_url="/")

        # Provide the next step
        if current_step < len(steps):
            step = steps[current_step]
            chat_history.append({"role": "assistant", "content": step})
            chat_history.append({
                "role": "assistant",
                "content": "Have you completed this step? (Yes/No)"
            })
            session["current_step"] = current_step  # Save the current step
        else:
            chat_history.append({
                "role": "assistant",
                "content": "Troubleshooting complete. If the issue persists, contact support."
            })

        # Save updated chat history in the session
        session["chat_history"] = chat_history

        return render_template_string(chatbot_page, model=model, chat_history=chat_history, back_url="/")

    else:
        # For all other queries, fetch information from the PDF
        if not stored_embeddings or not stored_chunks:
            chat_history.append({
                "role": "assistant",
                "content": "I'm sorry, I couldn't identify your problem. Please select a document to fetch relevant information."
            })
            session["chat_history"] = chat_history
            return render_template_string(chatbot_page, model=model, chat_history=chat_history, back_url="/")

        # Search the PDF for relevant content
        query_embedding = context_encoder.encode(user_query, convert_to_tensor=True, device=device).cpu().numpy()
        similarities = [
            torch.nn.functional.cosine_similarity(
                torch.tensor(query_embedding),
                torch.tensor(chunk_embedding),
                dim=0
            ).item()
            for chunk_embedding in stored_embeddings
        ]

        if similarities:
            most_relevant_index = similarities.index(max(similarities))
            most_relevant_chunk = stored_chunks[most_relevant_index]

            # Add the relevant chunk to the chat history
            chat_history.append({"role": "assistant", "content": most_relevant_chunk})
            session["chat_history"] = chat_history
            return render_template_string(chatbot_page, model=model, chat_history=chat_history, back_url="/")

        # If no relevant information is found
        chat_history.append({
            "role": "assistant",
            "content": "I couldn't find relevant information in the document. Please try a different query."
        })
        session["chat_history"] = chat_history
        return render_template_string(chatbot_page, model=model, chat_history=chat_history, back_url="/")

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return render_template_string(chatbot_page, model="Sample Model", chat_history=[], back_url="/select_model")

@app.route("/end_chat", methods=["POST"])
def end_chat():
    session.pop("chat_history", None)

    global stored_embeddings, stored_chunks, stored_metadata
    stored_embeddings = []
    stored_chunks = []
    stored_metadata = []

    return "<h1>Chat ended. </h1>", 200

def get_chunks(pdf_path):
    pdf_document = fitz.open(pdf_path)
    toc = pdf_document.get_toc()
    sections_list = []
    metadata = []  # To store page ranges and titles

    if toc:
        for i in range(len(toc)):
            title = toc[i][1]
            start_page = toc[i][2]
            end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else pdf_document.page_count
            section_text = extract_section(pdf_document, start_page, end_page, title)
            if section_text:
                sections_list.append(section_text)
                metadata.append({"title": title, "start_page": start_page, "end_page": end_page})
    else:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text("text")
            sections_list.append(page_text)
            metadata.append({"title": f"Page {page_num + 1}", "start_page": page_num + 1, "end_page": page_num + 1})

    pdf_document.close()
    return sections_list, metadata

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

if __name__ == "__main__":
    app.run(debug=True)


