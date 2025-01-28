import google.generativeai as genai
import pandas as pd
from keras.saving import load_model
import numpy as np
import ast

API_KEY = "apikey"
genai.configure(api_key=API_KEY)

diagnosis_model = load_model("./diagnosis.keras")
encoding = pd.read_csv('./dataset/disease_encoding.csv')
symptoms = pd.read_csv('./dataset/Symptom-severity.csv')
all_symptoms = symptoms['Symptom'].unique()


def get_diagonis(symptoms, all_symptoms = all_symptoms):
    patient_dict = {}
    for s in all_symptoms:
        if s in symptoms:
            patient_dict[s] = 1
        else:
            patient_dict[s] = 0
    
    patient_df = pd.DataFrame([patient_dict])
    prediction = diagnosis_model.predict(patient_df, verbose=0)
    diagnosis = encoding[encoding['Encoding'] == np.argmax(prediction)]['Disease'].values[0]

    return diagnosis

def get_symptoms(prompt, symptoms = all_symptoms):
    instructions = f"""You are a Doctor. Your task is to retrieve just symptoms and
    match them to this list: {symptoms}. Return just a python list of the symptoms and nothing else"""

    model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=instructions)

    response = model.generate_content(prompt)
    symptoms = ast.literal_eval(response.text)
    return symptoms

def chatbot():
    print("AI: Hello! Please indicate your symptoms.")

    messages = [
        {"role":"model","parts":"AI: Hello! Please indicate your symptoms."}
    ]

    instructions = output_instructions = f"""
You are an AI Doctor that uses a neural network model to diagnose a disease. 

If user gives you symptoms at any point in the conversation, no matter who the subject is, return just the word 'diagnose' as a string without anything else.

VERY IMPORTANT : Before any diagnosis or recommendations, 
always advise that your advice are not fully correct and seeking medical consult is recommended at the beginning of a recommendation.

Otherwise, The prompt that will be passed to you will be the diagnosis from the model. You will give the Diagnosis/disease and advise the recommended treatement or steps.
Do not repeat the symptoms. Be sure to follow the given structure of the message. After the initial diagnosis and advice, you are free to answer anything the user asks as long as it is related to his health.
But, If you are given any symptoms at any point of the conversation, regardless of the person, return just the word 'diagnose' as a string without anything else again.
Reject anything that is not related to medical field.
"""
    model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=instructions)
    while True:
        chat = model.start_chat(
            history=messages
        )
        q = input('User: ')

        if q.lower() == 'quit':
            break
        a = chat.send_message(q)

        if (a.text).strip('\n') == 'diagnose':
            user_symptoms = get_symptoms(q)
            diagnosis = get_diagonis(user_symptoms)
            print(diagnosis)
            a = chat.send_message(f"{q}. Based on this prompt, the model determined that the user has {diagnosis}. Respond to the user and present the diagnosis accordingly")

        print(f"AI: {a.text}")

        messages.append({"role":"user","parts":q})
        messages.append({"role":"model","parts":a.text})


def main():
    chatbot()

main()
