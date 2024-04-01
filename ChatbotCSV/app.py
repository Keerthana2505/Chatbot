import pandas as pd
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import csv
 
from keras.models import load_model
model = load_model('model.h5')
import json
import random
 
# Load intents from CSV into a list
intents_list = list(csv.reader(open('MasterData.csv', 'r', encoding='utf-8')))
 
# Load other necessary data
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
 
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)
 
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list
 
def getResponse(ints, intents_dict):
    tag = ints[0]['intent']
    result = 'Oops!! Try again'  # Initialize result with a default value
   
    if tag in intents_dict:
        responses = intents_dict[tag][3].split(',')
        print("Tag:", tag)
        print("Response:", responses)
        result = random.choice(responses)
       
    return result
 
def chatbot_response(msg):
    ints = predict_class(msg, model)
    print("Predicted ints:", ints)
    res = getResponse(ints, intents_dict)
    print("Res:", res)
   
    with open('datas.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        new_row = [msg, res]  # Replace with your actual data
        writer.writerow(new_row)
    return res
# Convert intents_list to a dictionary for efficient lookup
intents_dict = {i[1]: i for i in intents_list}


from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)
if __name__ == "__main__":
    app.run()