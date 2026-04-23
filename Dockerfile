FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords
COPY app.py .
COPY sentiment_model.pth .
COPY tfidf_vectorizer.pkl .
COPY label_encoder.pkl .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]