import os
import json
import openai
from PyPDF2 import PdfReader

# Initialize OpenAI client
client = openai.OpenAI(
    api_key="99ddc16a-6649-418b-880a-9cfdae4752da",
    base_url="https://api.openai.com/v1"
    )

# Define your PDF path and output directory
pdf_path = "C:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\Desktop\data\manuals\microsoft_surface_3.pdf" # 8 chunks got generated from this pdf with each chunk of chunk size 2k
output_dir = "output_json_chunks"  #output directory

def read_pdf_to_text(pdf_path):
    """
    Reads a PDF file and extracts the text content.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
 
def create_chunks(text, chunk_size):
    """
    Splits the text into chunks, where each chunk contains a specified number of words.
    Spaces are ignored in the chunk size calculation.
    """
    # Validate that chunk_size is an integer
    if not isinstance(chunk_size, int):raise TypeError(f"chunk_size must be an integer, but got {type(chunk_size).__name__}")
    print(f"chunk_size is of type: {type(chunk_size).__name__}")
 
    words = text.split()  # Split the text into words
    chunks = []
   
    for i in range(0, len(words), chunk_size):
        # Create a chunk containing `chunk_size` words
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        return chunks
 
     
 
 
def process_chunk_with_llm(chunk):
    """
    Creates a prompt for the LLM to convert each chunk into JSON format.
    """
    prompt = (
        f"""Convert the following chunks {chunk} into a well-structured JSON format.
        "Ensure all key-value pairs are meaningful:\n\n"
        {chunk}
        JSON format:
        [
            {{
                "name":"(model Name)",
                "version":"(version number or name)",
            }}
        ]
        """
    )
    #Sends the chunk to an LLM and returns the JSON response.
    try:
        response = client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",
            messages=[{"role": "system", "content": "You are a helpful assistant that converts chunk in JSON format."},
                     {"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return None
 
def save_json(data, output_dir, chunk_index):
    """
    Saves the JSON data to a file.
    """
    output_path = os.path.join(output_dir, f"chunk_{chunk_index}.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
 
def main(pdf_path,chunk_size=2000):
    """
    Main function to process a PDF, convert it to text, split it into chunks,
    process each chunk with an LLM, and save the JSON results.
    """
    # Step 1: Read PDF and extract text
    text = read_pdf_to_text(pdf_path)
 
    # Step 2: Split text into chunks
    chunks = create_chunks(text, chunk_size=2000)
 
    # # Step 3: Print each chunk
    for index, chunk in enumerate(chunks): #print(f"Chunk {index + 1}:\n{chunk}\n{'-'*50}") #This will print each chunk
        print(f"Processing chunk {index + 1}/{len(chunks)}...")
        json_response = process_chunk_with_llm(chunk)
        if json_response:
            try:
                json_data = json.loads(json_response)
                save_json(json_data, output_dir, index + 1)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for chunk {index + 1}")

if __name__ == "__main__":
    main(pdf_path,output_dir)