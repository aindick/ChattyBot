import random
import json
import pickle
import numpy as np
import nltk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


lenmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chattybotmodel.h5')

def clean_up(sentence):      # cleans up sentences
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lenmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):  # converts a sentence into a list of zeroes and ones that indicate if the word is there or not
    sentence_words = clean_up(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    res = [[i, r] for i, r in enumerate(result) if r > error_threshold]

    res.sort(key=lambda x: x[1], reverse= True)
    return_list = []
    for r in res:
        return_list.append({'intents': classes[r[0]], 'probability': str(r[1])})
    return return_list
def get_responses(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
print("Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_responses(ints, intents)
    print(res)