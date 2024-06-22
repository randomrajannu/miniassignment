# miniassignment

## Ollama Llama3.
Get up and running with large language models.
We tried to download it with windows with this link : https://ollama.com/download/OllamaSetup.exe

Run the setup on the file
![image](https://github.com/randomrajannu/miniassignment/assets/123664654/a017ef5f-e2cd-4ce6-acd3-72193567ee17)

Use Ollama with cmd to check what it offers.
![image](https://github.com/randomrajannu/miniassignment/assets/123664654/cf9d81bd-2b8f-4710-9eaf-c888d1b871a0)

The Ollama Serve code start the ollama server running in the local machine.
We have downloaded the LLama3 model and to run it. type in the code 

ollama run llama3.
![image](https://github.com/randomrajannu/miniassignment/assets/123664654/59885b94-55ef-4471-9e68-3415d940d012)

With the curl command we can check if it is running or not.

code:
curl http://127.0.0.1:11434/
![image](https://github.com/randomrajannu/miniassignment/assets/123664654/15815697-0637-4de3-927d-47ed3c1f198b)

We can also check the logs in the folder of Ollama, attaching to the repository.


## GPT-J, an open-source model by EleutherAI
### LLM Setup and Interaction

### Introduction
This repository contains setup instructions and scripts for deploying and interacting with Llama3 and GPT-J on a local machine.

### Prerequisites
- Python 3.x
- `pip` (Python package installer)
- Virtual environment tool (optional but recommended)

### Setup Instructions

#### 1. Environment Setup
1. Create and Activate Virtual Environment (optional but recommended)
   ```bash
   python -m venv llm_env
   source llm_env/bin/activate  # On Windows use `llm_env\Scripts\activate`

#### Install Required Packages

pip install flask torch transformers

. GPT-J Model Setup
Download and Install GPT-J

Download the model and tokenizer:
python
Copy code
from transformers import GPTJForCausalLM, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
Create Flask Application

python
Copy code
from flask import Flask, request, jsonify
from transformers import GPTJForCausalLM, GPT2Tokenizer

app = Flask(__name__)

Load the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

@app.route('/generate', methods=['POST'])
def generate_text():
    input_data = request.json
    if not input_data or 'text' not in input_data:
        return jsonify({"error": "Invalid input data"}), 400

    input_text = input_data.get("text", "")
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
#### 3. Run the Flask Server
Start the Flask Application
bash
Copy code
python app.py

#### 4. Interact with the Model via curl
Use curl Command to Generate Text
bash
Copy code
curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d '{"text": "Once upon a time"}'
