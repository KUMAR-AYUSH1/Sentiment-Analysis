# Import required libraries
import streamlit as st              
import pandas as pd                
import numpy as np                
import torch                       
import torch.nn as nn              
import joblib                      
import re            
import shap             
import string                      
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 

# 1. Cache resource loading so it only happens ONCE when the app starts
@st.cache_resource
def load_resources():
    # Load pre-trained TF-IDF vectorizer and label encoder
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # Initialize stopwords and stemmer
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    # Define Model Architecture
    class SentimentModel(nn.Module): 
        def __init__(self):
            super(SentimentModel, self).__init__()
            self.f1 = nn.Linear(150, 50)   
            self.f2 = nn.Linear(50, 20)    
            self.f3 = nn.Linear(20, 4)     

        def forward(self, x):
            x = torch.relu(self.f1(x))
            x = torch.relu(self.f2(x))
            return self.f3(x) 

    # Load trained PyTorch model weights
    model = SentimentModel()  
    model.load_state_dict(torch.load("sentiment_model.pth", map_location="cpu"))  
    model.eval()  

    # Define prediction function wrapper for SHAP
    def predict_func(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(x_tensor)
        return outputs.numpy()

    # FIX: Initialize the explainer natively in memory using empty/dummy data background
    # Since it's model-agnostic, a small reference frame (e.g., zero array) works perfectly
    dummy_background = np.zeros((1, 150))
    explainer = shap.Explainer(predict_func, dummy_background)
        
    return tfidf, le, stop_words, ps, model, explainer

# Unpack cached resources safely
tfidf, le, stop_words, ps, model, explainer = load_resources()
feature_names = tfidf.get_feature_names_out()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = ' '.join(tokens)
    text = [ps.stem(word) for word in tokens.split()]
    return ' '.join(text)

# Function to predict sentiment and generate SHAP analysis
def predict_sentiment(text):
    clean_text = preprocess_text(text)
    
    # Define the dense test vector correctly
    test_vector_dense = tfidf.transform([clean_text]).toarray()
    X_tensor = torch.tensor(test_vector_dense, dtype=torch.float32)  

    with torch.no_grad():
        output = model(X_tensor)  
        # Keep both the class index (for SHAP) and the text label
        predicted_class_idx = torch.argmax(output, dim=1).item()  
        predicted_label = le.inverse_transform([predicted_class_idx])[0]

    # Compute SHAP values dynamically
    shap_values = explainer(test_vector_dense)
    word_scores = shap_values.values[0, :, predicted_class_idx]

    activated_words = []
    for i, score in enumerate(word_scores):
        if test_vector_dense[0, i] > 0:
            activated_words.append((feature_names[i], score))

    # Compile and display word breakdown in UI
    if activated_words:
        df_importance = pd.DataFrame(activated_words, columns=['Stemmed Word', 'SHAP Impact Score'])
        df_importance = df_importance.sort_values(by='SHAP Impact Score', ascending=False)
        
        st.subheader("🔍 Word Contribution Breakdown")
        st.write(f"Showing features pushing the model toward **{predicted_label}**:")
        st.dataframe(df_importance, use_container_width=True)
    else:
        st.info("No vocabulary words from the model were found in this sentence.")

    return predicted_label

# Streamlit UI Setup
st.title("😐/🙁 Sentiment Analysis App")
st.write("Enter text to predict its sentiment (Normal, Depression, Suicidal, Stress).")

# Input text box
user_input = st.text_area("Enter your text here:", height=150)

# Button to trigger prediction
if st.button("Predict Sentiment", type="primary"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")
