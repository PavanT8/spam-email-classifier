# Spam Email Classifier

![Spam Email Classifier](https://img.shields.io/badge/Status-Active-brightgreen) ![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e.svg)

A professional, end-to-end Spam Email Classification system. This project includes a machine learning pipeline for processing text and training a model, alongside a FastAPI-based backend that serves predictions and a static frontend user interface.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Preprocessing](#preprocessing)
  - [Model](#model)
  - [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [API & UI](#api--ui)
- [Screenshots](#screenshots)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Problem Statement
Email spam is a persistent problem that degrades user experience and poses security risks. Identifying whether an incoming email is "spam" (unsolicited or malicious) or "ham" (legitimate) automatically helps reduce clutter and protect users. This project solves this by leveraging Natural Language Processing (NLP) and Machine Learning to classify text in real-time.

---

## Dataset
The project uses an open-source SMS/Email Spam Collection dataset. 
- **Structure**: The dataset contains two columns: `Category` (`ham` or `spam`) and `Message` (the text content).
- **Acquisition**: The training script automatically downloads the dataset if it is not present locally.

---

## Architecture
The repository is split into two primary components:
1. **Training Pipeline**: A Python script (`src/train.py`) that handles data acquisition, text preprocessing, model training, evaluation, and serialization (saving artifacts to the `models/` directory).
2. **Serving Backend**: A FastAPI application (`src/main.py`) that loads the serialized model and vectorizer to expose a RESTful prediction endpoint and serve a web frontend.

---

## Machine Learning Pipeline

### Preprocessing
Text data undergoes strict natural language processing using `NLTK`:
- **Noise Removal**: Non-alphabetic characters are stripped out.
- **Lowercasing**: Text is standardized to lowercase.
- **Stopwords Removal**: Common English stop words (e.g., "the", "a", "is") are removed.
- **Lemmatization**: Words are reduced to their base dictionary form (e.g., "running" becomes "run") using WordNetLemmatizer.

### Model
- **Vectorization**: `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) is used to convert the processed text into numerical features, capped at 5000 maximum features.
- **Algorithm**: A `MultinomialNB` (Multinomial Naive Bayes) classifier is trained on the TF-IDF features. Naive Bayes is highly effective and computationally efficient for text classification tasks.

### Evaluation
The dataset is split into an 80/20 train/test ratio. The model is evaluated on the test set using standard classification metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Installation

### Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### Steps
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd spam-email-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Train the Model
Before running the API, you need to train the model to generate the necessary artifacts (`classifier.pkl` and `vectorizer.pkl`):
```bash
python src/train.py
```
This will download the dataset, train the Naive Bayes model, print evaluation metrics to the console, and save the artifacts in the `models/` directory.

### 2. Start the Server
Once the models are saved, you can spin up the FastAPI server:
```bash
uvicorn src.main:app --reload
```
The server will run at `http://127.0.0.1:8000`.

---

## API & UI

### User Interface
Navigate to `http://127.0.0.1:8000/` in your browser. The backend serves a static `index.html` (if available in the `static/` directory) for users to interactively test email/message contents.

### REST API endpoint
- **Endpoint**: `POST /predict`
- **Request Body**:
  ```json
  {
    "text": "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now."
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "spam",
    "confidence": 0.985
  }
  ```

---

## Screenshots
> *(Placeholder for UI/API Screenshots)*

- **Web Interface**:
  <!-- ![Web UI Screenshot](path/to/screenshot.png) -->
- **API Response**:
  <!-- ![API Screenshot](path/to/api-screenshot.png) -->

---

## Limitations
- **Vocabulary Constraint**: The TF-IDF vectorizer is currently limited to 5000 features. Very rare or novel words might be ignored.
- **Context Ignorance**: Naive Bayes assumes conditional independence between words, which means it loses sequence and context information (unlike RNNs or Transformers).
- **Language**: The preprocessing is strictly designed for English text.

---

## Future Improvements
- **Advanced Models**: Integrate transformer-based architectures (e.g., BERT, RoBERTa) for better semantic understanding.
- **Hyperparameter Tuning**: Use `GridSearchCV` or `Optuna` to find the optimal vectorizer parameters and Naive Bayes alpha values.
- **Model Monitoring & Retraining**: Implement an endpoint to gather user feedback (misclassified emails) to continuously fine-tune the classifier.
- **Dockerization**: Create a `Dockerfile` and `docker-compose.yml` for simplified deployment.
