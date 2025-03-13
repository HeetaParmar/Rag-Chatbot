import fitz  # PyMuPDF for PDF to Image conversion
import os
import torch
import json
import re
from PIL import Image
# Define paths
# pdf_path = r"C:\\Users\\Heeta Parmar\\Downloads\\PO 5101023673_Galaxy Office Automation_HANA server AMC.pdf"
# image_output_dir = r"C:\Users\Heeta Parmar\Downloads\pdf_images"
# json_output_path = r"C:\Users\Heeta Parmar\Downloads\po_extracted_data.json"

# # Step 1: Convert PDF to Images using PyMuPDF
# os.makedirs(image_output_dir, exist_ok=True)
# doc = fitz.open(pdf_path)

# image_paths = []
# for page_num in range(len(doc)):
#     page = doc.load_page(page_num)  # Load page
#     pix = page.get_pixmap()  # Convert to an image
#     image_path = os.path.join(image_output_dir, f"page_{page_num + 1}.png")
#     pix.save(image_path)
#     image_paths.append(image_path)

# print("Images saved:", image_paths)
import os
import json
import requests
import base64
import pytesseract
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image

# Define paths
pdf_folder = r"C:\Users\Heeta Parmar\Downloads\PO 5101023673_Galaxy Office Automation_HANA server AMC.pdf"  # Folder containing PDFs
image_output_dir = r"C:\Users\Heeta Parmar\OneDrive - Galaxy Office Automation Pvt Ltd\Desktop\temp\images"  # Store images

os.makedirs(image_output_dir, exist_ok=True)  # Ensure the image output folder exists

# API endpoint (Replace with actual LLM API URL)
API_URL = "http://192.168.200.67:30502/process_image"

# Step 1: Convert PDFs to Images
image_paths = []
for pdf_file in os.listdir(pdf_folder):
    pdf_path = os.path.join(pdf_folder, pdf_file)

    if os.path.isfile(pdf_path) and pdf_file.lower().endswith(".pdf"):  # Ensure it's a valid PDF file
        print(f"🔄 Converting PDF: {pdf_file} to images...")

        try:
            images = convert_from_path(pdf_path, dpi=300)

            for i, image in enumerate(images):
                image_filename = f"{os.path.splitext(pdf_file)[0]}_page_{i+1}.png"
                image_path = os.path.join(image_output_dir, image_filename)
                image.save(image_path, "PNG")
                image_paths.append(image_path)
                print(f"✅ Saved Image: {image_path}")

        except Exception as e:
            print(f"❌ Error converting {pdf_file}: {e}")

print("✅ PDF to image conversion completed.")

# Step 2: Extract Text Using OCR
extracted_text = ""
for image_path in image_paths:
    try:
        print(f"🔍 Extracting text from {image_path} using OCR...")
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        extracted_text += text + "\n\n"

    except Exception as e:
        print(f"❌ Error in OCR for {image_path}: {e}")

print("✅ OCR Processing Complete.")

# Step 3: Define Questions for PO Extraction
questions = [
    "What is the Purchase Order (PO) number?",
    "What is the date of the Purchase Order?",
    "What is the billing address for the client?",
    "What is the delivery address or the service address of the buyer?",
    "What is the buyer's name in billing address?",
    "What is the buyer's name in delivery address or service address?",
    "What is the buyer pincode of billing address?",
    "What is the buyer pincode of delivery address or service address?",
    "What is the buyer state of billing address?",
    "What is the buyer state of delivery address or service address?",
    "What is the buyer contact number for billing?",
    "What is the buyer contact number for delivery?",
    "What is the buyer GST number for billing?",
    "What is the buyer GST number for delivery?",
    "What is the client PAN number for billing?",
    "What is the client PAN number for delivery?",
    "What is the email ID of the buyer?",
    "What is the email ID of the delivery person?",
    "What are the payment terms of the client?",
    "What is the general description from PO?",
    "What is the item number?",
    "What is the activity number?",
    "At what percentage of GST claimed?",
    "What is the total quantity?",
    "What is the unit rate in INR?",
    "What is the value in INR?",
    "What is the Activity number/SAC CODE?",
    "What is the total basic price?",
    "What is the CGST amount or percentage?",
    "What is the SGST amount or percentage?"
]

# Step 4: Call LLM API to Extract Information
def call_llm_api(question, extracted_text):
    try:
        payload = {
            "prompt": f"Extract the information from the given text (Purchase Order). Answer the following question: {question}. If the information is not present, respond with 'Not Found'. Return the result in JSON format.",
            "text_data": extracted_text  # Sending extracted text as input
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            return response.json()  # Assuming the API returns JSON
        else:
            print(f"❌ API Error: {response.status_code}, {response.text}")
            return {"question": question, "answer": "API Error"}

    except Exception as e:
        print(f"❌ API Request Failed: {e}")
        return {"question": question, "answer": "API Request Failed"}

# Extract Data from OCR Text
extracted_data = {}

for question in questions:
    response = call_llm_api(question, extracted_text)

    if "question" in response and "answer" in response:
        extracted_data[response["question"]] = response["answer"]
    else:
        extracted_data[question] = "Invalid API Response"

# Step 5: Print Extracted Data in JSON Format
print("\n📌 Extracted Information:")
print(json.dumps(extracted_data, indent=4))

# Step 6 (Optional): Convert First Image to Base64 and Send to API for Image Processing
if image_paths:
    with open(image_paths[0], "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "image_data": encoded_image,
        "prompt": "Extract structured information from the given purchase order image."
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        print("\n✅ API Image Processing Response:", response.json())
    else:
        print(f"\n❌ API Image Processing Error: {response.status_code}, {response.text}")
