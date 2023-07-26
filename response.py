import nltk
import json
import pickle
import random
import numpy as np
import tensorflow

ignore_words = ['?', '!',',','.', "'s", "'m"]

from data_preprocessing import get_stem_words

model=tensorflow.keras.models.load_model("C:/Users/DELL/Documents/Python programación/Clase/Chatbot/Clase120_Probandochatbot/chatbot_model.h5")
intents=json.loads(open("intents.json").read())
words=pickle.load(open("words.pkl", "rb"))
classes=pickle.load(open("classes.pkl", "rb"))

def processinputuser (inputuser):
    inputwordtoken1=nltk.word_tokenize(inputuser)
    inputwordtoken2=get_stem_words(inputwordtoken1, ignore_words)
    inputwordtoken2=sorted(list(set(inputwordtoken2)))
    bolsa=[]
    bagofwords=[]
    for palabra in words:
        if palabra in inputwordtoken2:
            bagofwords.append(1)
        else:
            bagofwords.append(0)
    bolsa.append(bagofwords)
    return np.array(bolsa)

def botclass(inputuser):
    inus=processinputuser(inputuser)
    prediction=model.predict(inus)
    labelprediction=np.argmax(prediction(0))
    return labelprediction

def botresponse(inputuser):
    classlabelpredict=botclass(inputuser)
    predictedclass=classes[classlabelpredict]
    for intent in intents["intents"]:
        if intent["tag"]==predictedclass:
            botresponse=random.choice(intent["responses"])
            return botresponse
print("Hola como te puedo ayudar?")

while True:
    inputuser=input("Escribe aqui")
    print("user: ", inputuser)
    response=botresponse(inputuser)
    print("Chatbot, reponde: ", response)