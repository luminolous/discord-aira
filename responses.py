import random
import json
import torch
from training.modulo import NeuralNet
from training.train import bag_of_words, tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_responses(user_input: str) -> str:
    lowered: str = user_input.lower()

    # Load GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    with open("intents.json", "r") as file:
        intents = json.load(file)

    while True:
        user_input = input("You: ")

        intent_tag = "unknown"
        for intent in intents["intents"]:
            for pattern in intent["patterns"]:
                if pattern.lower() in user_input.lower():
                    intent_tag = intent["tag"]
                    break
            if intent_tag != "unknown":
                break

        response = "I'm sorry, I didn't understand that."
        for intent in intents["intents"]:
            if intent["tag"] == intent_tag:
                response = torch.choice(intent["responses"])
                break

        print(f"{response}")


