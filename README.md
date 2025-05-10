# Medical AI Chatbot

## 📋 Overview

An intelligent medical assistant that helps users identify potential health conditions through:

- **Symptom-based diagnosis** using neural network classification
- **Treatment recommendations** powered by Google Gemini AI
- **Educational health information** with appropriate medical disclaimers

This project combines machine learning for disease prediction with generative AI for natural language understanding and personalized recommendations.

## 🏆 Key Features

- Natural language symptom extraction from user descriptions
- Multi-class neural network for disease classification 
- Severity-weighted symptom analysis for improved accuracy
- AI-generated treatment recommendations with medical context
- Clear medical disclaimers and professional consultation guidance

## 🛠️ Technical Implementation

### Dataset Processing

- **Source Data:**
  - Disease data (disease.csv) with binary symptom indicators
  - Pre-trained label encoder (label_encoder.pkl) for disease classification
  - 133 unique symptoms mapped to 41 disease categories

- **Input Processing:**
  - Natural language to standardized symptom mapping
  - Binary feature vector creation (1 for present symptoms, 0 for absent)
  - Direct vectorization for neural network input

### Machine Learning Architecture

- **Neural Network Model:**
  - Input layer: 133 nodes (one per symptom)
  - Hidden layers with dropout regularization
  - Softmax activation for multi-class prediction
  - Loss function: Categorical cross-entropy

### AI Integration Pipeline

1. **User Input Processing:**
   - Natural language symptom extraction via Google Gemini API
   - Symptom mapping to feature space

2. **Diagnosis Generation:**
   - Neural network prediction of most likely conditions
   - Confidence scoring for multiple potential diagnoses

3. **Recommendation Engine:**
   - Structured prompt engineering for Google Gemini
   - Context-aware treatment suggestions
   - Medical disclaimer integration



## ⚠️ Medical Disclaimer

This chatbot is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers for any medical concerns. This project is intended for educational and research purposes only.

## 📚 Technologies Used

- TensorFlow/Keras for neural network implementation
- Google Gemini API for natural language processing
- Python for data preprocessing and model training

## 🤝 Contributing

Contributions to improve the model accuracy, expand the dataset, or enhance the user experience are welcome. Please see the contribution guidelines before submitting pull requests.

## 📄 License

[License information]
