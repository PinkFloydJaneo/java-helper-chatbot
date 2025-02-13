from flask import Flask, render_template, request
import numpy as np
import random
import json
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

# Load the model and data
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

app = Flask(__name__)

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    return [word.lower() for word in sentence_words]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Debugging output
    print(f"Input: {sentence}")
    print(f"Results: {results}")
    
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(intents_list):
    tag = intents_list[0]["intent"]
    probability = intents_list[0]["probability"]  # Get the probability from the intents_list
    for i in intents["intents"]:
        if i["tag"] == tag:
            return [{"intent": tag, "response": random.choice(i["responses"]), "probability": probability}]
    return [{"intent": "unknown", "response": "Sorry, I couldn't find a response.", "probability": 0}]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_input = request.form['msg']
    intents_list = predict_class(user_input)
    
    # Check if intents_list is empty
    if not intents_list:
        return "Sorry, I didn't understand that."

    response = get_response(intents_list)
    
    # Debugging output
    print(f"Response: {response}")
    
    # Ensure response is a list
    if isinstance(response, list) and len(response) > 0:
        return f"{response[0]['response']}"
                # return f"Intent: {response[0]['intent']}, Probability: {response[0]['probability']}, Response: {response[0]['response']}"
    else:
        return "Sorry, I couldn't find a response."

if __name__ == "__main__":
    app.run(debug=True)