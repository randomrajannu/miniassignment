# miniassignment

# Ollama Llama3.
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

# Introduction to setting up GPT-Neo

GPT-Neo is an open-source implementation of EleutherAI's GPT-3 model, designed for natural language processing tasks such as textgeneration, completion, and more. This guide will walk you through the installation process to get GPT-Neo up and running on your local machine.

## Installation
Clone the Repository
Clone the GPT-Neo repository from GitHub to your local machine using the following command:

git clone https://github.com/EleutherAI/gpt-neo.git

cd gpt-neo

### make a python file and write the code in it.

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # Model name remains the same

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_conversation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

prompt = "Rajan: How is your health?\nHelly:"
conversation = generate_conversation(prompt)
print(conversation)

### desired output

![image](https://github.com/randomrajannu/miniassignment/assets/123664654/174bdfc1-633e-46ee-b196-f0e2d93bec87)



