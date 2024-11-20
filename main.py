import random
import json
import torch
import streamlit as st
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Initialize Streamlit
st.title("Sheriyans Coding School Chatbot 🤖")
st.write("Ask me anything about our courses, mentors, or services!")
bot_name = "Sheriyans Coding School"

# Maintain chat history using session state
if "history" not in st.session_state:
    st.session_state.history = []

# Input form
user_input = st.text_input("You:", placeholder="Type your message here...")

if user_input:
    # Process user input
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get model output
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Generate bot response
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
                break
    else:
        bot_response = "I do not understand..."

    # Add to chat history
    st.session_state.history.append(f"You: {user_input}")
    st.session_state.history.append(f"{bot_name}: {bot_response}")

# Display chat history
for message in st.session_state.history:
    st.write(message)
