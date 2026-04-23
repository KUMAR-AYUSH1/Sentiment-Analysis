# Sentiment-Analysis
Doing Sentiment Analysis NLP task 

to run docker image 
pull with
docker pull kumar2700/sentiment-app:latest
run with
docker run -p 8501:8501 kumar2700/sentiment-app


This project focuses on classifying mental health-related text into sentiment categories using NLP + Machine Learning + Deep Learning (ANN).
It also includes a Streamlit web app and Dockerized deployment.

Dataset
Source: Kaggle
Dataset: Sentiment Analysis for Mental Health
Size: ~30MB
🔗 Link: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

Project Workflow
🔹 Data Preprocessing
Text cleaning (lowercase, punctuation, numbers removal)
Stopword removal using NLTK
Stemming using PorterStemmer
Feature extraction using TF-IDF
Label encoding applied to target variable

👉 Simplification step:

Merged labels:
Anxiety, Bipolar, Personality Disorder → Stress


test 1: Feature Engineering
TF-IDF vectorization
Label encoding
Saved preprocessing artifacts (.pkl files)

test 2: Machine Learning Models
Logistic Regression
GradientBoostingClassifier
RandomForestClassifier
XGBClassifier

test 3: Deep Learning (ANN)
Built an Artificial Neural Network using PyTorch
Architecture:
Input → 150 features
Hidden layers → 50 → 20 neurons
Output → 4 classes
Achieved ~73% accuracy
Saved model as .pth
app.py = making streamlit app (preprocing,loading ann model

#docker file
Instead of COPY . .  
use
COPY app.py .
COPY sentiment_model.pth .
COPY tfidf_vectorizer.pkl .
COPY label_encoder.pkl .
