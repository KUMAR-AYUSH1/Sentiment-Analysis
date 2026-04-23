# Import required libraries
import streamlit as st              
import pandas as pd                
import numpy as np                
import torch                       
import torch.nn as nn              
import joblib                      
import re                         
import string                      
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 

# Load pre-trained TF-IDF vectorizer and label encoder
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

class SentimentModel(nn.Module): 
    def __init__(self):
        super(SentimentModel, self).__init__()
        
        # Fully connected layers
        self.f1 = nn.Linear(150, 50)   
        self.f2 = nn.Linear(50, 20)    
        self.f3 = nn.Linear(20, 4)     # Output layer (20 → 4 classes)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        return self.f3(x) 

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())

    # Tokenization (split into words)
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    tokens = ' '.join(tokens)

    # Apply stemming
    text = [ps.stem(word) for word in tokens.split()]
    text = ' '.join(text)

    return text

# Load trained PyTorch model weights
model = SentimentModel()  
model.load_state_dict(torch.load("sentiment_model.pth", map_location="cpu"))  # Load trained weights
model.eval()  # Set model to evaluation mode

# Function to predict sentiment
def predict_sentiment(text):
    text = preprocess_text(text)
    X = tfidf.transform([text])
    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)  # Convert to dense and then to tensor

    with torch.no_grad():
        output = model(X_tensor)  
        predicted_label = torch.argmax(output, dim=1).item()  # Get class index
        
        # Convert numeric label back to original label
        predicted_label = le.inverse_transform([predicted_label])[0]
    
    return predicted_label

# streamlit ui
st.title("😐/🙁 Sentiment Analysis App")
st.write("Enter text to predict its sentiment (Normal Depression Suicidal Stress ).")
# Input text box
user_input = st.text_area("Enter your text here:")

# Button to trigger prediction
if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter some text to analyze.")