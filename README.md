# Sentiment Analysis for Mental Health

An end-to-end Natural Language Processing (NLP) and Deep Learning project designed to classify mental health-related text into sentiment categories. The system utilizes an Artificial Neural Network (ANN) built with PyTorch, incorporates Explainable AI (XAI) using SHAP, and features an automated CI/CD pipeline via GitHub Actions to deploy a Streamlit application inside a Docker container.

---

## 🚀 Quick Start (Run via Docker)

You can run the pre-built application locally without setting up the Python environment.

**Pull the Docker Image:**
```bash
docker pull kumar2700/sentiment-app:latest
```

**Run the Application:**
```bash
docker run -p 8501:8501 kumar2700/sentiment-app:latest
```
Once running, open your web browser and navigate to `http://localhost:8501`.

---

## 📊 Dataset

* **Source:** Kaggle
* **Dataset Name:** Sentiment Analysis for Mental Health
* **🔗 Link:** [Kaggle Dataset URL](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

### Class Consolidation & Simplification
To optimize model performance, highly correlated overlapping states were merged into unified classifications:
* **Merged Targets:** *Anxiety*, *Bipolar*, and *Personality Disorder* were consolidated into a single label: **Stress**.
* **Final Target Classes (4):** `Normal`, `Depression`, `Suicidal`, `Stress`

---

## 🛠️ Project Workflow

### 🔹 Data Preprocessing
Standardized textual cleaning techniques applied via a custom preprocessing pipeline:
* Case folding (lowercasing text)
* Clean-up filters (URLs, numbers, and punctuation removal)
* Tokenization and stopword removal utilizing NLTK
* Word normalization via `PorterStemmer`
* Numerical vectorization utilizing a **TF-IDF Vectorizer** (configured for 150 feature outputs)

### 🔹 test1: Feature Engineering
* Generated TF-IDF matrix arrays.
* Encoded categorical labels using Scikit-Learn's `LabelEncoder`.
* Serialized processing instances cleanly to disk (`tfidf_vectorizer.pkl`, `label_encoder.pkl`).

### 🔹 test2: Machine Learning Evaluation
Evaluated multiple baseline traditional machine learning models to benchmark text classifications:
* Logistic Regression
* Gradient Boosting Classifier
* Random Forest Classifier
* XGBoost Classifier

### 🔹 test3: Deep Learning (ANN)
Developed an Artificial Neural Network (ANN) framework leveraging PyTorch.
* **Architecture:** 150 Input Features ➡️ Dense Layer (50, ReLU) ➡️ Dense Layer (20, ReLU) ➡️ Output Layer (4, Raw Logits)
* **Performance Metric:** Achieved **~73% Validation Accuracy**.
* **Storage:** Weights exported locally as `sentiment_model.pth`.

---

## 🔍 test4 Explainable AI (XAI) Integration
To interpret model assertions and address feature transparency, **SHAP (SHapley Additive exPlanations)** is integrated directly into the inference layer. Whenever a query is executed, the model computes word attribution scores. This allows users to see exactly which processed words drove the neural network toward its final target prediction.

---

## 🖥️ Application Files Overview

* **`app.py`**: A complete frontend web dashboard written in Streamlit. It handles user text collection, routes input strings through the preprocessing pipeline, loads saved model weights, calculates live SHAP scores, and renders interactive statistical dataframes.
* **`requirements.txt`**: Contains production Python framework dependencies mapped with performance flags (such as forcing CPU-only instances for PyTorch to optimize cloud deployment footprints).
* **`.github/workflows/docker.yml`**: Automates CI/CD code verification pipelines. On every code check-in to the main branch, it registers with Docker Hub, builds the container image target context, and builds a tag version snapshot.


<img width="748" height="832" alt="Screenshot 2026-07-09 023429" src="https://github.com/user-attachments/assets/c30e7f5b-770e-43a5-922d-499ddb4d8c4f" />
