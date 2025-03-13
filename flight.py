from PIL import Image
import io
from flask import Flask, request, render_template_string, session, url_for, send_from_directory, send_file
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
    api_key="2310ac5d-0f73-4359-a2dc-6ae2101eeb8d",
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
 
# Ensure static directories exist
if not os.path.exists("static/audio"):
    os.makedirs("static/audio")
if not os.path.exists("static/images"):
    os.makedirs("static/images")
 
# Initialize Sentence-Transformers model for embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2").to(device)
 
# Globals for storing embeddings and chunks
stored_embeddings = []
stored_chunks = []
 
 
# Load Sentence-Transformers model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2").to(device)
 
def get_template(title, content):
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
        <title>{title}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background: linear-gradient(120deg, #ff0000, #ffffff, #0000ff);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .container {{
                background-color: rgba(255, 255, 255, 0.8);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                width: 60%;
            }}
        </style>
    </head>
    <body>
        <div class="container text-center">
            {content}
        </div>
    </body>
    </html>
    """
 
login_page = get_template("Login to Galaxy Office Automation", """
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
""")
 
model_page = get_template("Select Model", """
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
""")
 
company_page = get_template("Select Company", """
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
""")
 
model_list_page = get_template("Select Model", """
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
""")
chatbot_page = get_template("Chat with Model", """
    <h1 class="my-5">Welcome to Galaxy Help Desk</h1>
    <div class="border p-4 bg-white shadow-sm rounded mb-4">
        <h4>Chat History:</h4>
        <ul class="list-group">
            {% for message in chat_history|reverse %}
                <li class="list-group-item chat-message">
                    <strong>{{ message.role.capitalize() }}:</strong> {{ message.content }}
                    {% if message.images and message.images|length > 0 %}
                        <div class="images-container mt-2">
                            {% for image_path in message.images %}
                                <div class="image-item mb-2">
                                    <p>Image: {{ image_path }}</p>
                                    <img src="{{ url_for('static', filename=image_path) }}" 
                                         alt="Image from PDF" class="img-fluid" style="max-width: 100%; max-height: 300px;">
                                </div>
                            {% endfor %}
                        </div>
                    {% endif %}
                    {% if message.audio %}
                        <div class="audio-controls mt-2">
                            <audio id="audio-{{ loop.index }}" src="{{ url_for('static', filename='audio/' + message.audio) }}"></audio>
                            <button onclick="togglePlay('audio-{{ loop.index }}')" class="btn btn-sm btn-primary">Play/Pause</button>
                            <select onchange="changeSpeed('audio-{{ loop.index }}', this.value)" class="form-select-sm d-inline-block w-auto">
                                <option value="0.5">0.5x</option>
                                <option value="1.0" selected>1.0x</option>
                                <option value="1.5">1.5x</option>
                                <option value="2.0">2.0x</option>
                            </select>
                        </div>
                    {% endif %}
                    {% if "Have you performed this step? [Yes] [No]" in message.content %}
                        <div class="mt-2">
                            <form method="POST" action="/step_response" class="d-inline">
                                <input type="hidden" name="model" value="{{ model }}">
                                <button type="submit" name="step_response" value="yes" class="btn btn-success btn-sm me-2">Yes</button>
                                <button type="submit" name="step_response" value="no" class="btn btn-danger btn-sm">No</button>
                            </form>
                        </div>
                    {% endif %}
                </li>
            {% endfor %}
        </ul>
    </div>
    <script>
        function togglePlay(audioId) {
            const audio = document.getElementById(audioId);
            if (audio.paused) {
                audio.play();
            } else {
                audio.pause();
            }
        }
 
        function changeSpeed(audioId, speed) {
            const audio = document.getElementById(audioId);
            audio.playbackRate = parseFloat(speed);
        }
        
        function startRecognition() {
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                const recognition = new SpeechRecognition();
                
                recognition.lang = 'en-US';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;
                
                recognition.start();
                
                recognition.onresult = function(event) {
                    const speechResult = event.results[0][0].transcript;
                    document.getElementById('question').value = speechResult;
                };
                
                recognition.onerror = function(event) {
                    console.error('Speech recognition error:', event.error);
                    alert('Speech recognition error: ' + event.error);
                };
                
                recognition.onend = function() {
                    console.log('Speech recognition ended');
                };
            } else {
                alert('Speech recognition is not supported in your browser. Please use Chrome or Edge.');
            }
        }
    </script>
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
    <div class="d-flex justify-content-between">
        <a href="/" class="btn btn-secondary">Back</a>
        <form method="POST" action="/clear_chat" class="d-inline">
            <button type="submit" class="btn btn-warning me-2">Clear Chat</button>
        </form>
        <form method="POST" action="/end_chat" class="d-inline">
            <button type="submit" class="btn btn-danger">End Chat</button>
        </form>
    </div>
""")
 
@app.route("/static/<path:filename>")
def serve_audio(filename):
    return send_from_directory("static", filename)
 
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
 
# Helper function for short greetings
GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
 
def is_greeting(text):
    return text.lower().strip() in GREETINGS
 
# Decision Tree Troubleshooting
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
 
def find_decision_tree(question_embedding):
    best_similarity = -1.0
    best_issue = None
    for issue, data in decision_trees.items():
        for tag in data["tags"]:
            tag_embedding = context_encoder.encode(tag, convert_to_tensor=True, device=device).cpu().numpy()
            similarity = np.dot(question_embedding, tag_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(tag_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_issue = issue
    return best_issue if best_similarity >= 0.7 else None
 
from markupsafe import Markup
def format_response_as_steps(text):
    """Formats text to display each sentence in a new line."""
    steps = text.split(".")
    formatted_text = "<br>".join([f"• {step.strip()}" for step in steps if step.strip()])
    return Markup(formatted_text)

def make_hyperlink(text):
    """Converts URLs, numbers, and image paths into clickable blue hyperlinks."""
    import re
    text = re.sub(r'(https?://\S+)', r'<a href="\1" style="color:blue;">\1</a>', text)
    text = re.sub(r'\b(\d+)\b', r'<a href="#" style="color:blue;">\1</a>', text)
    text = re.sub(r'(static/images/\S+\.(?:png|jpg|jpeg|gif))', r'<a href="\1" style="color:blue;">\1</a>', text)
    return Markup(text)

def generate_audio(text):
    """Generates an audio file for the given text."""
    audio_filename = f"audio_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join("static/audio", audio_filename)
    tts = gTTS(text=text, lang='en')
    tts.save(audio_path)
    return audio_filename

@app.route("/ask_question", methods=["POST"])
def ask_question():
    global stored_embeddings, stored_chunks
    model = request.form["model"]
    question = request.form.get("question", "").strip()
    
    if "chat_history" not in session or not session.get("chat_history"):
        session["chat_history"] = [{"role": "system", "content": "How may I help you?"}]
    
    if not question:
        return "<h1>Error: No valid input provided.</h1>", 400
    
    system_message = "System: Processing your request...<br>"
    
    if is_greeting(question):
        answer_text = f"{question.capitalize()}, how may I help you?"
        answer_audio = generate_audio(answer_text)
        answer_images = []
    else:
        if not stored_embeddings or not stored_chunks:
            return "<h1>Error: No embeddings available. Please select a document first.</h1>", 400
        
        question_embedding = context_encoder.encode(question, convert_to_tensor=True, device=device).cpu().numpy()
        best_issue = find_decision_tree(question_embedding)
        
        if best_issue:
            session["decision_tree"] = {
                "issue": best_issue,
                "steps": decision_trees[best_issue]["steps"],
                "current_step": 0,
                "question": question
            }
            step_text = f"Troubleshooting Step 1: {decision_trees[best_issue]['steps'][0]}<br>Have you performed this step? [Yes] [No]"
            answer_text = make_hyperlink(step_text)
            answer_audio = generate_audio(step_text)
            answer_images = []
        else:
            similarities = [
                torch.nn.functional.cosine_similarity(
                    torch.tensor(question_embedding, dtype=torch.float),
                    torch.tensor(entry["embedding"], dtype=torch.float),
                    dim=0
                ).item() for entry in stored_embeddings
            ]
            
            if not similarities or max(similarities) < 0.3:
                fallback_prompt = f"User asked: '{question}'. Please provide a detailed answer in a stepwise manner."
                response = client.chat.completions.create(
                    model="Meta-Llama-3.1-8B-Instruct",
                    messages=[{"role": "system", "content": "You are an expert assistant."},
                              {"role": "user", "content": fallback_prompt}],
                    temperature=0.4,
                    top_p=0.1,
                )
                answer_text = format_response_as_steps(response.choices[0].message.content)
                answer_images = []
            else:
                most_relevant_index = similarities.index(max(similarities))
                most_relevant_chunk = stored_chunks[most_relevant_index]
                answer_text = format_response_as_steps(most_relevant_chunk["text"])
                
                # Get images from the most relevant chunk
                answer_images = most_relevant_chunk.get("images", [])
                
                # Debug image paths
                print(f"Images found in chunk: {answer_images}")
                
                # If there are no images in the most relevant chunk, check other chunks with high similarity
                if not answer_images and len(similarities) > 1:
                    # Get top 3 most similar chunks
                    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
                    for idx in top_indices:
                        if idx != most_relevant_index:  # Skip the most relevant chunk we already checked
                            chunk_images = stored_chunks[idx].get("images", [])
                            if chunk_images:
                                answer_images = chunk_images
                                print(f"Using images from another chunk: {answer_images}")
                                break
            
            answer_audio = generate_audio(answer_text)

    # Add user's question to chat history
    chat_history = session.get("chat_history", [])
    chat_history.append({"role": "user", "content": question})
    
    # Add system's response to chat history
    chat_history.append({"role": "assistant", "content": answer_text, "audio": answer_audio, "images": answer_images})
    session["chat_history"] = chat_history
    
    return render_template_string(chatbot_page, chat_history=chat_history, model=model, back_url="/select_model")
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
    
    # Generate audio for the response
    audio_filename = generate_audio(message)
    
    conversation_history = session.get("chat_history", [{"role": "system", "content": "How may I help you?"}])
    conversation_history.append({"role": "assistant", "content": message, "audio": audio_filename, "images": []})
    session["chat_history"] = conversation_history
    
    return render_template_string(chatbot_page, model=request.form.get("model"), chat_history=conversation_history, back_url="/", decision_tree=dt)
@app.route("/view_pdf", methods=["GET"])
def view_pdf():
    user_id = request.args.get("user_id")
    user = next((u for u in user_data if str(u["user_id"]) == user_id), None)
    
    if user:
        pdf_path = user.get("manual_location")
        if pdf_path and os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=False)
        else:
            return "Error: Manual PDF not found.", 404
    return "Error: User not found.", 404


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

def extract_pdf_images(pdf_path, output_dir="static/images"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            img_filename = f"page_{page_num + 1}_img_{img_index}.{img_ext}"
            img_path = os.path.join(output_dir, img_filename)
            image.save(img_path)
            images.append(img_path)
    return images

def get_chunks(pdf_path):
    try:
        # Try the standard way first
        pdf_document = fitz.open(pdf_path)
    except AttributeError:
        # If that fails, try the alternative import method
        import pymupdf
        pdf_document = pymupdf.open(pdf_path)
    except Exception as e:
        # If both fail, try a different approach with PyMuPDF
        from pymupdf import Document
        pdf_document = Document(pdf_path)
    
    sections_list = []
    image_dir = "static/images"
    os.makedirs(image_dir, exist_ok=True)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        toc = pdf_document.get_toc()
        text = page.get_text("text")
        image_paths = []
        
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                img_ext = base_image["ext"]
                
                # Create a unique filename for the image
                img_filename = f"page_{page_num+1}_img_{img_index+1}.{img_ext}"
                img_filepath = os.path.join(image_dir, img_filename)
                
                # Save the image to disk
                with open(img_filepath, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Store the relative path to be used in the template
                # Use only the path relative to the static directory with forward slashes
                relative_path = "images/" + img_filename
                image_paths.append(relative_path)
                print(f"Extracted image: {img_filepath} -> URL path: {relative_path}")
            except Exception as e:
                print(f"Error extracting image: {e}")
                continue
        
        chunk_data = {
            "text": text.strip(),
            "images": image_paths,
            "page": page_num + 1
        }
        sections_list.append(chunk_data)
    
    pdf_document.close()
    return sections_list

def process_pdf_and_generate_embeddings(pdf_path, batch_size=50):
    chunks = get_chunks(pdf_path)  # Extract chunks from PDF
    
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("No valid chunks found! Ensure PDF text extraction works properly.")
    
    embeddings = []
    formatted_chunks = []
    
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or "text" not in chunk or not chunk["text"].strip():
            print(f"Skipping invalid chunk at index {index}: {chunk}")  # Debugging info
            continue  # Skip empty or malformed chunks
        
        stepwise_text = f"Q: {chunk['text']}\nA:"  # Keep original text, format answer in steps
        formatted_chunks.append(stepwise_text)
        
        embedding = context_encoder.encode(chunk["text"], convert_to_tensor=True, device=device)
        embeddings.append({
            "embedding": embedding.cpu().numpy(),
            "images": chunk.get("images", []),
            "page": chunk.get("page", "Unknown")
        })
    
    if not embeddings:
        raise ValueError("Embeddings could not be generated. Check extracted text content.")
    
    rephrased_steps = []
    
    for i in range(0, len(formatted_chunks), batch_size):
        batch = formatted_chunks[i:i + batch_size]
        rephrase_prompt = """
        The following are extracted steps from a document. Please format them so that each sentence appears on a new line using  /n without modifying the original text.
        
        Steps:
        """ + "\n".join(batch) + "\n\nProvide the formatted steps as requested."
        
        response = client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",
            messages=[{"role": "system", "content": "You are an expert in technical documentation formatting."},
                      {"role": "user", "content": rephrase_prompt}],
            temperature=0.4,
            top_p=0.1,
        )
        
        rephrased_steps.append(response.choices[0].message.content)
    
    final_rephrased_text = "\n".join(rephrased_steps)
    print("Reformatted Steps:\n", final_rephrased_text)
    
    return embeddings, final_rephrased_text


def convert_chunks_to_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = context_encoder.encode(chunk["text"], convert_to_tensor=True, device=device)
        
        # Ensure image paths are using forward slashes
        image_paths = []
        if "images" in chunk and chunk["images"]:
            for img_path in chunk["images"]:
                # Convert backslashes to forward slashes if present
                img_path = img_path.replace("\\", "/")
                image_paths.append(img_path)
        
        embeddings.append({
            "embedding": embedding.cpu().numpy(), 
            "images": image_paths, 
            "page": chunk.get("page", "Unknown")
        })
    return embeddings

@app.route("/debug_images")
def debug_images():
    """Debug route to check image paths and display."""
    image_dir = "static/images"
    if not os.path.exists(image_dir):
        return f"Image directory {image_dir} does not exist", 404
    
    images = os.listdir(image_dir)
    if not images:
        return f"No images found in {image_dir}", 404
    
    html = "<h1>Debug Images</h1>"
    html += f"<p>Found {len(images)} images in {image_dir}</p>"
    html += "<div>"
    
    for img in images[:10]:  # Show first 10 images
        img_path = "images/" + img
        img_url = url_for("static", filename=img_path)
        
        html += f'<div style="margin: 20px; padding: 10px; border: 1px solid #ccc;">'
        html += f'<p>File: {img}</p>'
        html += f'<p>Path: {img_path}</p>'
        html += f'<p>URL: {img_url}</p>'
        html += f'<img src="{img_url}" style="max-width: 300px; border: 1px solid #ccc;">'
        html += '</div>'
    
    html += "</div>"
    
    return html

@app.route("/debug_chunks")
def debug_chunks():
    """Debug route to check stored chunks and their image paths."""
    global stored_chunks
    
    if not stored_chunks:
        return "No chunks stored. Please select a document first.", 400
    
    html = "<h1>Debug Chunks</h1>"
    html += f"<p>Total chunks: {len(stored_chunks)}</p>"
    
    # Count chunks with images
    chunks_with_images = sum(1 for chunk in stored_chunks if chunk.get("images") and len(chunk["images"]) > 0)
    html += f"<p>Chunks with images: {chunks_with_images}</p>"
    
    # Display first 5 chunks with images
    count = 0
    for i, chunk in enumerate(stored_chunks):
        if "images" in chunk and chunk["images"]:
            count += 1
            html += f'<div style="margin: 20px; padding: 10px; border: 1px solid #ccc;">'
            html += f'<h3>Chunk {i} (Page {chunk.get("page", "Unknown")})</h3>'
            html += f'<p><strong>Text:</strong> {chunk["text"][:200]}...</p>'
            html += f'<p><strong>Images ({len(chunk["images"])}):</strong></p>'
            html += '<div style="display: flex; flex-wrap: wrap;">'
            
            for img_path in chunk["images"]:
                img_url = url_for("static", filename=img_path)
                html += f'<div style="margin: 10px; padding: 10px; border: 1px solid #ddd;">'
                html += f'<p>Path: {img_path}</p>'
                html += f'<p>URL: {img_url}</p>'
                html += f'<img src="{img_url}" style="max-width: 200px; border: 1px solid #ccc;">'
                html += '</div>'
            
            html += '</div></div>'
            
            if count >= 5:
                break
    
    if count == 0:
        html += "<p>No chunks with images found.</p>"
    
    return html

# Add a new route to directly view a specific image
@app.route("/view_image/<path:image_path>")
def view_image(image_path):
    """Route to directly view a specific image."""
    return send_from_directory("static", image_path)

@app.route("/test_image")
def test_image():
    """Test route to display a specific image from the static/images directory."""
    image_dir = "static/images"
    if not os.path.exists(image_dir):
        return f"Image directory {image_dir} does not exist", 404
    
    images = os.listdir(image_dir)
    if not images:
        return f"No images found in {image_dir}", 404
    
    # Get the first image
    test_image = images[0]
    img_path = "images/" + test_image
    img_url = url_for("static", filename=img_path)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Test</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .image-container {{ border: 1px solid #ccc; padding: 10px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Image Test</h1>
        <p>Testing image display from static directory</p>
        
        <div class="image-container">
            <h3>Image Details:</h3>
            <p>Filename: {test_image}</p>
            <p>Path: {img_path}</p>
            <p>URL: {img_url}</p>
            
            <h3>Direct Image Tag:</h3>
            <img src="{img_url}" style="max-width: 300px; border: 1px solid #ccc;">
            
            <h3>Alternative Image Tag:</h3>
            <img src="/static/{img_path}" style="max-width: 300px; border: 1px solid #ccc;">
        </div>
        
        <p><a href="/debug_images">View All Debug Images</a></p>
    </body>
    </html>
    """
    
    return html

if __name__ == "__main__":
    app.run(debug=True)
 
 