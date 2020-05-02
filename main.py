import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)

print(data["intents"])

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words_tokenized = nltk.word_tokenize(pattern)
        words.extend(words_tokenized)

print(words)
