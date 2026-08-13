import os
import pandas as pd
import urllib.request
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Setup
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download('stopwords', download_dir=NLTK_DATA_DIR)
nltk.download('wordnet', download_dir=NLTK_DATA_DIR)

DATA_URL = 'https://raw.githubusercontent.com/codebasics/py/master/ML/14_naive_bayes/spam.csv'
DATA_PATH = 'data/dataset.csv'
MODELS_DIR = 'models'
os.makedirs('data', exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def acquire_data():
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    else:
        print("Dataset already exists.")
    df = pd.read_csv(DATA_PATH)
    # The dataset has 'Category' and 'Message'
    df.rename(columns={'Category': 'label', 'Message': 'text'}, inplace=True)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def main():
    print("Acquiring data...")
    df = acquire_data()
    
    print("Preprocessing data...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    X = df['processed_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training TF-IDF and Model...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    
    print("Evaluating Model...")
    y_pred = classifier.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    print("Saving model and vectorizer...")
    joblib.dump(classifier, os.path.join(MODELS_DIR, 'classifier.pkl'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer.pkl'))
    print("Pipeline complete.")

if __name__ == '__main__':
    main()
