from PIL import Image
import io
from flask import Flask, request, render_template_string, session, url_for
import json
import os
import fitz  # PyMuPDF for extracting text from PDFs
from sentence_transformers import SentenceTransformer
import torch
import openai
from gtts import gTTS  # for text-to-speech
import uuid
from werkzeug.utils import secure_filename
import numpy as np  # for cosine similarity calculations

# Initialize OpenAI client
client = openai.OpenAI(
    api_key="0b10c39d-6e9a-453b-afa2-8d1a2b0a44f2",
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
            color: #333;
        }
        .chat-container {
            background-color: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 400px;
        }
        .highlight { font-weight: bold; color: #007bff; }
        .btn-primary, .btn-secondary { border-radius: 50px; }
        input[type="text"] {
            border-radius: 50px;
            padding: 10px 15px;
        }
        .chat-history {
            max-height: 300px;
            overflow-y: auto;
        }
        /* Preserve newlines in messages */
        .chat-message { white-space: pre-wrap; }
    </style>
</head>
<body class="bg-light">
    <div class="chat-container">
        <h1 class="text-center my-5">Welcome to Galaxy Help Desk </h1>
        <!-- Chat History (Latest messages at the top) -->
        <div class="border p-4 bg-white shadow-sm rounded mb-4">
            <h3>Chat History:</h3>
            <ul class="list-group">
                {% for message in chat_history|reverse %}
                    <li class="list-group-item chat-message">
                        <strong>{{ message.role.capitalize() }}:</strong> {{ message.content }}
                    </li>
                {% endfor %}
            </ul>
        </div>
        <!-- Form to Ask a New Question -->
        <form method="POST" action="/ask_question" class="d-flex flex-column align-items-center mb-3">
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
        <!-- Dynamic Troubleshooting Session (if active) -->
        {% if decision_tree %}
        <div class="alert alert-info mt-3">
            <p><strong>Troubleshooting for: {{ decision_tree.issue }}</strong></p>
            <p>Your question: {{ decision_tree.question }}</p>
            <p>Step {{ decision_tree.current_step + 1 }}: {{ decision_tree.steps[decision_tree.current_step] }}</p>
            <form method="POST" action="/step_response">
                <input type="hidden" name="model" value="{{ model }}">
                <button type="submit" name="step_response" value="yes" class="btn btn-success">Yes</button>
                <button type="submit" name="step_response" value="no" class="btn btn-danger">No</button>
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

# Routes
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
                chunks = get_chunks(pdf_path)
                embeddings = convert_chunks_to_embeddings(chunks)
                global stored_embeddings, stored_chunks
                stored_embeddings = embeddings
                stored_chunks = chunks
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
    chunks = get_chunks(pdf_path)
    embeddings = convert_chunks_to_embeddings(chunks)
    global stored_embeddings, stored_chunks
    stored_embeddings = embeddings
    stored_chunks = chunks
    return render_template_string(chatbot_page, model=model, chat_history=[], back_url="/company_page")

@app.route("/ask_question", methods=["POST"])
def ask_question():
    global stored_embeddings, stored_chunks
    model = request.form["model"]
    question = request.form.get("question")
    if not question:
        return "<h1>Error: Please enter a question.</h1>", 400

    # Convert question to embedding
    question_embedding = context_encoder.encode(question, convert_to_tensor=True, device=device).cpu().numpy()

    # --- Decision Tree Matching ---
    decision_trees = {
        "display issue": {
            "steps": [
                "Check if the display is powered on.",
                "Verify the input source.",
                "Check for loose cables.",
                "Restart the display and computer.",
                "Test with another device.",
                "Contact support."
            ],
            "tags": ["screen", "display", "monitor", "visual", "power"]
        },
        "network issue": {
            "steps": [
                "Check the network cable.",
                "Restart the router.",
                "Verify the Wi-Fi connection.",
                "Contact your ISP."
            ],
            "tags": ["internet", "network", "connection", "wifi", "router"]
        }
    }
    best_similarity_dt = -1.0
    best_issue_dt = None
    best_tag_dt = None
    for issue, data in decision_trees.items():
        for tag in data["tags"]:
            tag_embedding = context_encoder.encode(tag, convert_to_tensor=True, device=device).cpu().numpy()
            similarity_dt = np.dot(question_embedding, tag_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(tag_embedding)
            )
            if similarity_dt > best_similarity_dt:
                best_similarity_dt = similarity_dt
                best_issue_dt = issue
                best_tag_dt = tag

    DECISION_THRESHOLD = 0.7
    PDF_THRESHOLD = 0.3

    if best_issue_dt is not None and best_similarity_dt >= DECISION_THRESHOLD:
        # Start a dynamic troubleshooting session.
        session["decision_tree"] = {
            "issue": best_issue_dt,
            "steps": decision_trees[best_issue_dt]["steps"],
            "current_step": 0,
            "question": question
        }
        full_answer = ("Your question: " + question + "\n" +
                       "Troubleshooting Step 1: " + decision_trees[best_issue_dt]["steps"][0] +
                       "\nHave you performed this step? [Yes] [No]")
        answer = full_answer
    else:
        # Try PDF retrieval.
        if not stored_embeddings:
            return "Error: No document loaded.", 400
        similarities = [
            torch.nn.functional.cosine_similarity(
                torch.tensor(question_embedding, dtype=torch.float),
                torch.tensor(entry["embedding"], dtype=torch.float),
                dim=0
            ).item() for entry in stored_embeddings
        ]
        if not similarities or max(similarities) < PDF_THRESHOLD:
            # Fallback: generate answer from LLM with a prompt.
            fallback_prompt = f"User asked: '{question}'. Please provide a detailed answer based on your general knowledge."
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[{"role": "system", "content": "You are an expert assistant."},
                          {"role": "user", "content": fallback_prompt}],
                temperature=0.4,
                top_p=0.1,
            )
            answer = response.choices[0].message.content
            full_answer = answer
        else:
            most_relevant_index = similarities.index(max(similarities))
            most_relevant_chunk = stored_chunks[most_relevant_index]
            relevant_images = stored_embeddings[most_relevant_index]["images"]
            image_text = "\n".join(relevant_images)
            conversation_history = session.get("chat_history", [{"role": "system", "content": "How may I help you?"}])
            conversation_history.append({"role": "user", "content": question})
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=conversation_history + [{"role": "assistant", "content": most_relevant_chunk["text"]}],
                temperature=0.4,
                top_p=0.1,
            )
            answer = response.choices[0].message.content
            full_answer = (f"{answer}\n"
                           f"Page: {most_relevant_chunk['page']}\n"
                           f"Images:\n{image_text}\n")
    conversation_history = session.get("chat_history", [{"role": "system", "content": "How may I help you?"}])
    conversation_history.append({"role": "assistant", "content": full_answer})
    session["chat_history"] = conversation_history
    decision_tree = session.get("decision_tree", None)
    audio_filename = f"static/audio_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=answer, lang='en')
    tts.save(audio_filename)
    return render_template_string(chatbot_page, model=model, chat_history=conversation_history, back_url="/", decision_tree=decision_tree)

@app.route("/step_response", methods=["POST"])
def step_response():
    dt = session.get("decision_tree", None)
    if not dt:
        return "No troubleshooting session active.", 400
    response_val = request.form.get("step_response")
    if response_val == "yes":
        dt["current_step"] += 1
        if dt["current_step"] >= len(dt["steps"]):
            message = "All troubleshooting steps completed. If your issue persists, please contact support."
            session.pop("decision_tree", None)
        else:
            message = ("Troubleshooting Step " + str(dt["current_step"] + 1) + ": " +
                       dt["steps"][dt["current_step"]] +
                       "\nHave you performed this step? [Yes] [No]")
    else:
        current = dt["current_step"]
        message = ("Please perform the following step before proceeding:\n" +
                   dt["steps"][current] +
                   "\nOnce completed, please click Yes.")
    conversation_history = session.get("chat_history", [{"role": "system", "content": "How may I help you?"}])
    conversation_history.append({"role": "assistant", "content": message})
    session["chat_history"] = conversation_history
    return render_template_string(chatbot_page, model=request.form.get("model"), chat_history=conversation_history, back_url="/", decision_tree=dt)

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return render_template_string(chatbot_page, model="Sample Model", chat_history=[], back_url="/select_model")

@app.route("/end_chat", methods=["POST"])
def end_chat():
    session.pop("chat_history", None)
    global stored_embeddings, stored_chunks
    stored_embeddings = []
    stored_chunks = []
    return "<h1>Chat ended.</h1>", 200

def extract_section(pdf_document, start_page, end_page, title, output_dir="static/pdf_images"):
    section_text = f"Section Title: {title}\n\n"
    images = []
    os.makedirs(output_dir, exist_ok=True)
    for page_num in range(start_page - 1, end_page):
        page = pdf_document.load_page(page_num)
        section_text += page.get_text("text") + "\n"
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            img_filename = f"page_{page_num+1}_img_{img_index}.{img_ext}"
            img_path = os.path.join(output_dir, img_filename)
            image.save(img_path)
            images.append(img_path)
    return {"text": section_text, "images": images}

def get_chunks(pdf_path):
    pdf_document = fitz.open(pdf_path)
    sections_list = []
    image_dir = "static/images"
    os.makedirs(image_dir, exist_ok=True)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text = page.get_text("text")
        image_paths = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(pdf_document, xref)
            if pix.n < 4:
                rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
            else:
                rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
                pix = None
            img_filename = f"page_{page_num+1}_img_{img_index+1}.png"
            img_filepath = os.path.join(image_dir, img_filename)
            rgb_pix.save(img_filepath)
            rgb_pix = None
            image_paths.append(img_filepath)
        chunk_data = {
            "text": text.strip(),
            "images": image_paths,
            "page": page_num + 1
        }
        sections_list.append(chunk_data)
    pdf_document.close()
    return sections_list

def convert_chunks_to_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = context_encoder.encode(chunk["text"], convert_to_tensor=True, device=device)
        embeddings.append({"embedding": embedding.cpu().numpy(), "images": chunk["images"], "page": chunk["page"]})
    return embeddings

if __name__ == "__main__":
    app.run(debug=True)
