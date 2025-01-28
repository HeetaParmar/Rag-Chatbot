import json
import openai
import os
import requests

client = openai.OpenAI(
    api_key="272ae1ae-0157-40c5-9ab8-dcacd29e56f8",
    base_url="https://api.sambanova.ai/v1",
)


# Prompt for generating JSON data
def get_ai_response(user_input):
    prompt= f"""
    You are an intelligent assistant. Generate data like name, arrays of multiple hobbies, gender either Male or Female, and Birth Date YYYY-MM-DD,
    and store responses in JSON format. Respond politely to salutations and greetings. You will also save the previously generated chat details and upon asking by user you will answer that previous chat answer to user.
    Generate the response strictly in the following format:
    {{
        "Name": any random name,
        "Hobbies": a list of 1-4 hobbies,
        "Gender": one of "Male", "Female", or "Other",
        "BirthDate": a randomly generated date in the format "YYYY-MM-DD"
    }}
    """

# Call Meta-Llama model for response
response = client.chat.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that generates JSON data."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.1,
    top_p=0.1
)

def generate_json_from_prompt():
    # Get user input for the prompt
    print("Enter your prompt to generate JSON data:")
    user_prompt = input("> ")

    # Meta-Llama model setup and call
    try:
        response = client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates JSON data."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            top_p=0.1
        )
        
        # Parse the response to extract generated content
        generated_json = response.choices[0].message.content

        # Save the JSON data to a file
        json_file_path = "c:/Users/Heeta Parmar/OneDrive - Galaxy Office Automation Pvt Ltd/Desktop/generated_data_meta_llama.json"
        with open(json_file_path, "w") as json_file:
            json_file.write(generated_json)

        print(f"JSON data successfully generated and saved at: {json_file_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Main execution
if __name__ == "__main__":
    generate_json_from_prompt()
