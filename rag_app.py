from flask import Flask, request, render_template, jsonify, session
from rag_all_functions import load_json, get_chunks, convert_chunks_to_embeddings, save_chunks_to_file
import openai
import os 

# Initialize Flask app
app = Flask(__name__)
# app.secret_key = "supersecretkey"  # Replace with a secure key in production

# Initialize OpenAI client
client = openai.OpenAI(
    api_key="559480d3-5c94-47ed-b1e1-4d9a30fb51f9",
    base_url="https://api.sambanova.ai/v1",
)

# Load user data and manuals
user_data = load_json("user_data.json")
manuals_data = load_json("manual_data.json")

# Globals for storing embeddings and chunks
stored_embeddings = []
stored_chunks = []

@app.route("/", methods=["GET"])
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    user = next((user for user in user_data if str(user["user_id"]) == username), None)

    if user:
        if "password" in user and user["password"] == password:
            return render_template("model.html", username=username, model=user["model"])
        elif "password" not in user:
            return render_template("model.html", username=username, model=user["model"])
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

                return render_template("chatbot.html", model=model, chat_history=[])
            else:
                return f"<h1>Error:</h1><p>PDF for the model {model} not found or does not exist at '{pdf_path}'.</p>", 404
    else:
        companies = sorted(set(manual["company"] for manual in manuals_data))
        return render_template("company.html", username=username, companies=companies)

@app.route("/select_model", methods=["POST"])
def select_model():
    username = request.form["username"]
    company = request.form["company"]

    models = [manual["model"] for manual in manuals_data if manual["company"] == company]
    return render_template("model_list.html", username=username, models=models)

@app.route("/ask_question", methods=["POST"])
def ask_question():
    global stored_embeddings, stored_chunks

    model = request.form["model"]
    question = request.form["question"]

    if not stored_embeddings or not stored_chunks:
        return "<h1>Error: No embeddings available. Please select a document first.</h1>", 400

    # Generate embedding for the user query
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

    response = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",
        messages=conversation_history,
        temperature=0.1,
        top_p=0.1,
    )

    answer = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})

    session["chat_history"] = conversation_history

    return render_template("chatbot.html", model=model, chat_history=conversation_history)

@app.route("/end_chat", methods=["POST"])
def end_chat():
    session.pop("chat_history", None)
    global stored_embeddings, stored_chunks
    stored_embeddings = []
    stored_chunks = []

    return "<h1>Chat ended. You can start a new session.</h1>", 200

if __name__ == "__main__":
    app.run(debug=True)

