from PIL import Image
import io
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
import fitz  # PyMuPDF

 
# Initialize OpenAI client
client = openai.OpenAI(
    api_key="4631532b-ba03-441e-bb60-4c7086a5f228",
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
                        <strong>{{ message.role.capitalize() }}:</strong> {{ message.content }}
                        {% if message.role == 'assistant' and message.audio_file %}
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

    # Compute cosine similarity with stored embeddings
    similarities = [
        torch.nn.functional.cosine_similarity(
            torch.tensor(question_embedding),
            torch.tensor(entry["embedding"]),
            dim=0
        ).item()
        for entry in stored_embeddings
    ]

    if not similarities:
        return "<h1>Error: No relevant results found.</h1>", 400

    # Get the most relevant chunk
    most_relevant_index = similarities.index(max(similarities))
    most_relevant_chunk = stored_chunks[most_relevant_index]

    # Get relevant images
    relevant_images = stored_embeddings[most_relevant_index]["images"]
    image_tags = "".join(f'<img src="{img}" style="max-width:100%;" class="img-thumbnail">' for img in relevant_images)

    # Generate OpenAI response
    conversation_history = session.get("chat_history", [{"role": "system", "content": "How may I help you?"}])
    conversation_history.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",
        messages=conversation_history + [{"role": "assistant", "content": most_relevant_chunk["text"]}],
        temperature=0.4,
        top_p=0.1,
    )

    answer = response.choices[0].message.content

    # Convert response to speech
    tts = gTTS(text=answer, lang="en")
    audio_filename = f"static/audio/{uuid.uuid4()}.mp3"
    os.makedirs("static/audio", exist_ok=True)  # Ensure directory exists
    tts.save(audio_filename)

    # Append response, images, and audio to chat history
    conversation_history.append({
        "role": "assistant",
        "content": f"{answer}<br><strong>Page: {most_relevant_chunk['page']}</strong><br>{image_tags}",
        "audio_file": audio_filename
    })
    session["chat_history"] = conversation_history

    # Render chatbot page with audio support
    return render_template_string(chatbot_page, model=model, chat_history=conversation_history, back_url="/")
 
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
 
    return "<h1>Chat ended. </h1>", 200
def extract_section(pdf_document, start_page, end_page, title, output_dir="static/pdf_images"):
    """Extracts text and images from a PDF section."""
    section_text = f"Section Title: {title}\n\n"
    images = []
 
    os.makedirs(output_dir, exist_ok=True)
 
    for page_num in range(start_page - 1, end_page):
        page = pdf_document.load_page(page_num)
        section_text += page.get_text("text") + "\n"
 
        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
 
            # Save image
            img_filename = f"page_{page_num+1}_img_{img_index}.{img_ext}"
            img_path = os.path.join(output_dir, img_filename)
            image.save(img_path)
 
            # Store image reference
            images.append(img_path)
 
    return {"text": section_text, "images": images}
 

def get_chunks(pdf_path):
    pdf_document = fitz.open(pdf_path)
    sections_list = []
    image_dir = "static/images"  # Store extracted images
    os.makedirs(image_dir, exist_ok=True)
 
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text = page.get_text("text")
 
        # Extract images
        image_paths = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]  # Reference ID of image
            pix = fitz.Pixmap(pdf_document, xref)
 
            # If the image is not in RGB, convert it to RGB
            if pix.n < 4:  # If the image is grayscale or RGB
                rgb_pix = fitz.Pixmap(fitz.csRGB, pix)  # Convert to RGB
            else:
                # If the image is CMYK, convert to RGB
                rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
                pix = None  # Free original Pixmap memory
 
            # Save the image as PNG
            img_filename = f"page_{page_num+1}_img_{img_index+1}.png"
            img_filepath = os.path.join(image_dir, img_filename)
            rgb_pix.save(img_filepath)  # Save the image in RGB format
            rgb_pix = None  # Free memory for Pixmap
 
            image_paths.append(img_filepath)
 
        # Combine text and image references
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
 
 