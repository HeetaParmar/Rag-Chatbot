import json
import os
import fitz  # PyMuPDF for extracting text from PDFs
from sentence_transformers import SentenceTransformer
import torch

# Load JSON files
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Load SentenceTransformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2").to(device)

# Extract sections from PDF
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

# Convert chunks to embeddings
def convert_chunks_to_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = context_encoder.encode(chunk, convert_to_tensor=True, device=device)
        embeddings.append(embedding.cpu().numpy())
    return embeddings

# Save chunks to a file
def save_chunks_to_file(chunks, filename="chunks.json"):
    with open(filename, 'w') as f:
        json.dump(chunks, f)

# Load chunks from a file
def load_chunks_from_file(filename="chunks.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []
