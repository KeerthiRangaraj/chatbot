import json
import random
import nltk
import pyttsx3
import threading
import gradio as gr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load intents from the updated JSON file
with open('intents.json') as f:
    data = json.load(f)

# Ensure NLTK tokenizers are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("all")

# Prepare the training data
X, y = [], []
for intent in data['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

# Vectorize the patterns and train the Naive Bayes classifier
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X_vec = vectorizer.fit_transform(X)

clf = MultinomialNB()
clf.fit(X_vec, y)

# Initialize pyttsx3 voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 175)  # Set speed
engine.setProperty('volume', 1.0)  # Max volume

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[Voice Error]: {e}")

# Main function to get chatbot response with voice output
def get_response(user_input):
    vec = vectorizer.transform([user_input])
    pred = clf.predict(vec)[0]

    for intent in data['intents']:
        if intent['tag'] == pred:
            response = random.choice(intent['responses'])
            threading.Thread(target=speak, args=(response,)).start()
            return response

    fallback = "I'm not sure I understand. Can you rephrase?"
    threading.Thread(target=speak, args=(fallback,)).start()
    return fallback

# Gradio Interface
iface = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(lines=1, placeholder="Say something..."),
    outputs="text",
    title="🗣️ Voice-Enabled ChatBot",
    description="Chat with an AI bot that talks back! Type anything and hear it respond."
)

iface.launch()
