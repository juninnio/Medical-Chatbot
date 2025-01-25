# Medical AI Chatbot Project
Overview
-------------------------
This project is an ongoing development of a medical AI chatbot aimed at assisting users by:

- Diagnosing diseases based on symptoms provided by the user.
- Offering advice or recommendations for treatment based on the diagnosis.
- Educating users about health conditions while emphasizing the importance of consulting a medical professional for accurate advice.
The chatbot uses machine learning and neural network models for diagnosis, integrates symptom extraction using Google Gemini API, and provides user-friendly interactions with generative AI for advice and recommendations.

Current Progress
-------------------------
### 1. Dataset Preparation
Cleaned and processed medical datasets that include:
- Symptom severity (Symptom-severity.csv)
- Disease-to-symptom mappings (dataset.csv)
- One-hot encoded symptoms for modeling purposes.
- Weighted symptom severity to improve model predictions.
### 2. Machine Learning Model
Built a neural network model with TensorFlow/Keras to perform multi-class classification.<br>
Incorporated:
- Input layers for 133 symptoms.
- Dropout layers to prevent overfitting.
- Softmax activation for multi-class output.
- Achieved satisfactory training and test performance with proper regularization.
### 3. Symptom Extraction
- Integrated Google Gemini API to extract symptoms from natural language input provided by users.
- Developed prompts to ensure symptoms are correctly mapped to the model’s input feature space.
### 4. Diagnosis and Recommendation Pipeline
- Diagnosed diseases based on user-input symptoms using the trained neural network model.
- Generated treatment advice or recommendations using Google Gemini API with a structured prompt.

Future Implementations
-------------------------------
### 1. Incorporate Patient Medical History
Enable the model to consider previous diagnoses or patient history to improve predictions.
### 2. Self-Learning and Feedback
Implement mechanisms for the model to improve over time using user feedback. <br>
Example: Reinforcement learning techniques to fine-tune recommendations.
### 3. Expand Dataset
Increase the dataset size with more diseases, symptoms, and severity levels. <br>

### 4. User-Friendly Chatbot Interface
Develop a front-end for seamless user interaction.<br>
Options:
- Web-based chatbot.
- Mobile application integration.

### 5. Enhanced Explainability
Provide users with insights into how the diagnosis was made.
### 6. Safety Features
Include disclaimers emphasizing that the AI is not a substitute for professional medical advice.<br>
Ensure the chatbot routes users to emergency services or professionals for critical symptoms.

Disclaimer
--------------------
This chatbot is not a substitute for professional medical advice. Always consult a doctor for medical concerns or emergencies. This project is a proof-of-concept and is intended for educational purposes.
