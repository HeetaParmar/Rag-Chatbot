from flask import Flask, request, render_template_string, jsonify
import json
import os
import fitz  # PyMuPDF for extracting text from PDFs

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

# Login Page Template
login_page = """
<!doctype html>
<html lang="en">
<head>
    <title> Welcome to Galaxy Office Automation </title>
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

# Model List Template
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

@app.route("/", methods=["GET"])
def home():
    return render_template_string(login_page)

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    # Convert user_id to string for comparison
    user = next((user for user in user_data if str(user["user_id"]) == username), None)

    if user:
        if "password" in user and user["password"] == password:
            return render_template_string(model_page, username=username, model=user["model"])
        elif "password" not in user:  # If no password is stored, skip password validation
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
        # Show the PDF from the location stored in user data
        user = next((user for user in user_data if str(user["user_id"]) == username), None)
        if user:
            pdf_path = user.get("pdf_path")  # Get the stored PDF path for this user
            if pdf_path and os.path.exists(pdf_path):
                pdf_text = extract_pdf_text(pdf_path)
                return f"""
                    <h1>Welcome, user {username}!</h1>
                    <h2>You have selected the model: {model}.</h2>
                    <h3>Extracted PDF Content:</h3>
                    <pre>{pdf_text[:500]}...</pre>
                    <p><i>(Only the first 500 characters are displayed)</i></p>
                """
            else:
                return f"<h1>Error:</h1><p>PDF for the model {model} not found or the file does not exist at '{pdf_path}'.</p>", 404
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

    # Find the manual for the selected model
    manual = next((manual for manual in manuals_data if manual["model"] == model), None)

    if manual and os.path.exists(manual["location"]):
        # Extract text from the PDF
        pdf_text = extract_pdf_text(manual["location"])
        if pdf_text.strip():  # Ensure the PDF text is not empty
            return f"""
                <h1>Welcome, user {username}!</h1>
                <h2>You have selected the model: {model}.</h2>
                <h3>Extracted PDF Content:</h3>
                <pre>{pdf_text[:500]}...</pre>
                <p><i>(Only the first 500 characters are displayed)</i></p>
            """
        else:
            return f"<h1>Welcome, user {username}!</h1><h2>You have selected the model: {model}.</h2><p>Could not extract text from the PDF.</p>"
    else:
        return f"<h1>Error:</h1><p>Manual for model {model} not found or the file does not exist.</p>", 404

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

if __name__ == "__main__":
    app.run(debug=True)
