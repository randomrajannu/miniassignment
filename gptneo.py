# Import the necessary classes from the transformers library
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the model name
model_name = "gpt2"  # Model name remains the same

# Load the GPT-2 tokenizer and model using the specified model name
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define a function to generate a conversation based on the given prompt
def generate_conversation(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the conversation using the model
    outputs = model.generate(
        inputs.input_ids, 
        max_length=200, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    
    # Decode the generated text into a readable format
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Define the prompt for the conversation
prompt = "Rajan: How is your health?\nHelly:"

# Generate the conversation based on the prompt
conversation = generate_conversation(prompt)

# Print the generated conversation
print(conversation)
