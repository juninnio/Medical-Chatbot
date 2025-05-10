from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timedelta
import os
import logging
from keras import saving
from google import genai
from google.genai import types
import pandas as pd
import numpy as np
import pickle

client = genai.Client(api_key="gemini api")
model = 'gemini-2.0-flash'

diagnosis_model = saving.load_model('disease_model.keras')
df = pd.read_csv('disease.csv').drop('diseases', axis=1)
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

contents = []

class ParseUserInput(BaseModel):
    matched_symptoms: list[str] = Field(description="List of matched symptoms")

def parse_user_input(user_input):
    """
    Parse the user input to extract symptoms.
    """

    instructions = f"""You are a medical assistant. Given the user's input symptoms, return the list of matching standardized symptom names.

    User input: '{user_input}'

    Match the input to this list of possible symptoms:{df.columns}"""

    response = client.models.generate_content(
        model=model,
        contents=user_input,
        config={
            "response_mime_type": 'application/json',
            "response_schema": ParseUserInput,
            'system_instruction': instructions
        }
    )
    
    return eval(response.text)['matched_symptoms']

def prepare_data(cols):
    input_data = {col: 1 if col in cols else 0 for col in df.columns}
    input_df = pd.DataFrame([input_data])
    vals = input_df.values

    return vals

def get_prediction(symptoms):
    inputs = prepare_data(symptoms)
    prediction = diagnosis_model.predict(inputs)
    predicted_class = np.argmax(prediction[0])
    predicted_disease = le.inverse_transform([predicted_class])[0]
    
    return predicted_disease

class RouteUserInput(BaseModel):
    input_type: Literal["new_diagnosis","follow-up", "other"] = Field(
        description="""Type of user's input. new_diagnosis if user wants make a new diagnosis, 
        follow-up if user has follow-up questions or general medical questions,
        other if user request does not match any of the categories.""")


def main():
    instruction  = """You are a compassionate and knowledgeable AI Medical Assistant designed to provide informative, accurate responses to medical inquiries. Your primary goals are to:

1. Explain medical diagnoses in clear, accessible language that patients can understand
2. Answer follow-up questions about medical conditions, treatments, and general health concerns
3. Provide evidence-based information while acknowledging the limits of your knowledge
4. Respect that you are not a replacement for professional medical care

When explaining a diagnosis:
- Be clear and confident in the diagnosis.
- Describe the condition in simple terms, avoiding excessive medical jargon
- Explain common symptoms, causes, and typical progression of the condition
- Outline standard treatment approaches and their purposes
- Emphasize the importance of following their doctor's specific advice

When answering follow-up questions:
- Provide factual, up-to-date medical information based on established medical consensus
- Include relevant context that might help the user better understand their situation
- Be honest about limitations in your knowledge and encourage professional consultation when appropriate
- Never discourage seeking proper medical care or following a doctor's advice

Important limitations to acknowledge:
- You cannot access patient records or personal medical history
- You cannot interpret test results or imaging studies
- You should encourage users to consult healthcare professionals for specific medical advice

Use a warm, empathetic tone that acknowledges health concerns can be stressful, while maintaining professionalism and accuracy in all responses.

Remember: Your goal is to inform and educate, not to replace professional medical care."""
    config = types.GenerateContentConfig()
    print('Welcome to AI Medical Assistant')
    print('I am here to help you diagnose based on your symptoms.')
    print('To ensure a precise diagnosis, please give me as many of your symptoms as you can.\n')

    while True:
        user_input = input("User: ")

        if user_input == 'exit':
            break

        input_response = client.models.generate_content(
            model=model,
            contents=user_input,
            config={
                'response_mime_type': 'application/json',
                'response_schema': RouteUserInput
            }
        )
        route = eval(input_response.text)['input_type']

        contents.append(types.Content(
            role='user', parts=[types.Part(text=user_input)]
        ))
        print(route)

        if route == "new_diagnosis":
            parsed_input = parse_user_input(user_input)
            print(parsed_input)
            prediction = get_prediction(parsed_input)
            response = client.models.generate_content(
                model=model,
                contents=f"Based on {user_input}, the predicted diagnosis is: {prediction}",
                config=config
            )

            output = response.text

        elif route == "follow-up":
            response = client.models.generate_content(
                model=model,
                contents=user_input,
                config=config
            )

            output = response.text

        elif route == 'other':
            response = client.models.generate_content(
                model=model,
                contents=user_input,
                config={
                    'system_instruction':"You are a medical assistant, reject the user in a respectful way"
                }
            )

            output = response.text

        print(f"Model: {response.text}")
        contents.append(types.Content(
            role='model', parts=[types.Part(text=response.text)]
        ))

main()  

