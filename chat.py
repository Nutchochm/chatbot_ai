
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    # Define model paths
    quantized_model_path = "Models/quantized-typhoon-8b-base"
    base_model_path = "Models/typhoon-8b-base"

    # Load the model architecture
    model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, local_files_only=True)

    # Load the quantized model weights
    state_dict = torch.load(f"{quantized_model_path}/pytorch_model.bin", map_location=device)

    # Filter state_dict to match model architecture if necessary
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

    # Load the weights into the model with strict=False to allow mismatched keys
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    # Log missing and unexpected keys for debugging
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    # Set the model to evaluation mode and move to device
    model.eval()
    model.to(device)

    # Ensure the pad_token_id is properly set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id if not already set

    print("Model and tokenizer loaded successfully.")


def gen_response(query):
    global model, tokenizer
    print(f"User : {query}")
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must be loaded before generating responses. Call load_model() first.")

    inputs = tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=200
    )
    # Move inputs to device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print(f"Inputs to device : {inputs}")

    # Perform inference with attention mask
    #with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
        eos_token_id=tokenizer.eos_token_id,
        num_beams=1
    )
    print(f"Output : {outputs}")
    # Decode the generated tokens into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"AI : {generated_text}")
    return generated_text

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break        
        print("AI is generating a response...")
        load_model()
        resp = gen_response(sentence)
        print(resp)


#AI คืออะไร

"""import random
import json
import os
import time
import torch
import numpy as np
import bitsandbytes as bnb
from flask import jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from nltk import word_tokenize as eng_tokenize
from langdetect import detect
from pythainlp.tokenize import word_tokenize as thai_tokenize
"""
"""
def tokenize(sentence):
    lang = detect(sentence)
    if lang == 'th':  # Thai
        return thai_tokenize(sentence, engine='newmm')
    else:  # Default to English
        return eng_tokenize(sentence)
    
def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [w.lower() for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
"""
"""
def find_json_files(directory):
    # Return a list of JSON files in the specified directory
    return [f for f in os.listdir(directory) if f.endswith('.json')]

def find_model_files(directory):
    # Return a list of model files in the specified directory
    return [f for f in os.listdir(directory) if f.endswith('.pth')]

# Path for JSON files
jspath = "pretrained/"
jsfiles = find_json_files(jspath)

# Print and process JSON files
for jsfile in jsfiles:
    print(f"Found JSON file: {jsfile}")
    full_js_path = os.path.join(jspath, jsfile)
    with open(full_js_path, 'r', encoding='utf-8') as json_data:
        intents = json.load(json_data)
    print(f"Loaded intents from {jsfile}")

# Path for model files
modelpath = "pretrained/model/"
model_files = find_model_files(modelpath)

# Print and process model files
for mdfile in model_files:
    print(f"Found model file: {mdfile}")
    full_model_path = os.path.join(modelpath, "intent_20241205.pth")
    # Load the model file with torch
    data = torch.load(full_model_path)
    print(f"Loaded model from {mdfile}")
"""
"""
jsfile = 'pretrained/intent_20241205.json'
with open(jsfile, 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)
    print(f"Loaded intents from {jsfile}")


FILE = "pretrained/model/intent_20241205.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
responses_map = data['responses_map']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()



def get_response(msg, language=""):
    # Tokenize and process the message
    tokens = tokenize(msg)
    X = bag_of_words(tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Predict using the trained model
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    print(f"Predicted tag: {tag}  |  token : {tokens}")
    print(f"output: {output}  |  predicted item : {predicted.item()}")

    # Calculate probabilities
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    print(f"Prediction probability: {prob.item()}")

    if prob.item() > 0.75:
        # Match the tag with intents
        if tag in responses_map:
            # Check for responses in the desired language, fallback to English
            responses = responses_map[tag].get(language, responses_map[tag].get(language, []))
            if responses:
                return random.choice(responses)
        elif responses is None:
            loading_effect()
            return generate_response(responses)
    
    return "i do not understand..."

# Load the model and tokenizer
#model_id = "scb10x/llama-3-typhoon-v1.5-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
)


# Function to generate responses
def generate_response(query):
    messages = [
        {"role": "system", "content": "You are a helpful assistant who always speaks Thai."},
        {"role": "user", "content": query},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llm_model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = llm_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)
"""
