import json
print(json.__file__)  # This will print the location of the json module being used

# Example data (Python dictionary)
data = {
    "name": ["Alice","xyz","abc"],
    "age": [25,56,78],
    "city": ["New York","Los Angeles","Chicago"],
    "hobbies": ["reading", "traveling", "coding"]
}

# 1. Convert Python dictionary to JSON string
json_string = json.dumps(data, indent=4)  # Convert to JSON format
print("JSON String:\n", json_string)

# 2. Write JSON data to a file
with open("data.json", "w") as json_file:
    json.dump(data, json_file, indent=4)  # Use json.dump() to save to file

# 3. Read JSON data from a file
with open("data.json", "r") as json_file:
    loaded_data = json.load(json_file)  # Load from file
    print("\nLoaded Data:\n", loaded_data)

# Access a specific value
print("\nCity:", loaded_data["city"])  # Access a specific value
