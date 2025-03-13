from flask import Flask, render_template_string, request, session
import PyPDF2
import re

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    text_chunks = []
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text_chunks.append(page.extract_text())
    return text_chunks

# Load the PDF data
pdf_path = "troubleshooting_guide.pdf"
stored_chunks = [{"text": chunk} for chunk in extract_text_from_pdf(pdf_path)]

# Function to fetch troubleshooting steps based on the issue
def get_troubleshooting_steps(issue):
    steps = []
    for chunk in stored_chunks:
        if issue.lower() in chunk["text"].lower():
            # Split steps by new lines or bullet points
            steps = re.split(r"\n•|\n\d+\.|\n-", chunk["text"])
            steps = [step.strip() for step in steps if step.strip()]
            break
    return steps if steps else ["No troubleshooting steps found for this issue."]

# Function to extract agent contact details
def get_agent_contact():
    for chunk in stored_chunks:
        if "contact" in chunk["text"].lower() or "support" in chunk["text"].lower():
            match = re.search(r'\+?\d{1,3}[-.\s]?\d{2,3}[-.\s]?\d{3}[-.\s]?\d{4}', chunk["text"])
            return match.group(0) if match else "N/A"
    return "N/A"

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model = request.form["model"]
        issue = request.form["issue"]

        troubleshooting_steps = get_troubleshooting_steps(issue)

        session["troubleshooting_steps"] = troubleshooting_steps
        session["current_step"] = 0

        return render_template_string(
            troubleshooting_page,
            model=model,
            step=troubleshooting_steps[0],
            step_number=1,
            message="",
            back_url="/"
        )

    return render_template_string(home_page)

# Next Step Route
@app.route("/next_step", methods=["POST"])
def next_step():
    model = request.form["model"]
    action = request.form["action"]

    if "troubleshooting_steps" not in session or "current_step" not in session:
        return "<h1>Error: No active troubleshooting session.</h1>", 400

    steps = session["troubleshooting_steps"]

    if action == "no":
        return render_template_string(
            troubleshooting_page,
            model=model,
            step=steps[session["current_step"]],
            step_number=session["current_step"] + 1,
            message="⚠️ Please complete the above step before proceeding.",
            back_url="/"
        )

    # Move to the next step
    session["current_step"] += 1

    if session["current_step"] >= len(steps):
        agent_number = get_agent_contact()
        return f"<h1>Issue not resolved? Kindly contact an agent at {agent_number}.</h1>", 200

    return render_template_string(
        troubleshooting_page,
        model=model,
        step=steps[session["current_step"]],
        step_number=session["current_step"] + 1,
        message="",
        back_url="/"
    )

# Home Page Template
home_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Troubleshooting Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container text-center">
        <h1 class="my-5">Troubleshooting Assistant</h1>
        <form method="POST">
            <div class="mb-3">
                <label for="model" class="form-label">Enter Device Model:</label>
                <input type="text" id="model" name="model" class="form-control" required>
            </div>
            <div class="mb-3">
                <label for="issue" class="form-label">Describe Your Issue:</label>
                <input type="text" id="issue" name="issue" class="form-control" required>
            </div>
            <button type="submit" class="btn btn-primary">Start Troubleshooting</button>
        </form>
    </div>
</body>
</html>
"""

# Troubleshooting Page Template
troubleshooting_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Troubleshooting Guide</title>
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
        <h1 class="my-5">Troubleshooting Guide</h1>
        <p><strong>Step {{ step_number }}:</strong> {{ step }}</p>

        {% if message %}
            <p class="text-danger">{{ message }}</p>
        {% endif %}

        <form method="POST" action="/next_step" class="d-inline-block">
            <input type="hidden" name="model" value="{{ model }}">
            <button type="submit" name="action" value="yes" class="btn btn-success">Yes, Next Step</button>
            <button type="submit" name="action" value="no" class="btn btn-danger">No</button>
        </form>
        <div class="mt-4">
            <a href="{{ back_url }}" class="btn btn-secondary">Back</a>
        </div>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
