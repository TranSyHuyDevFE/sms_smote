# SMS Detection Pipeline Example (Simple Demo with SVM + RF Ensemble)
# Author: Huy Tran - Based on "Malicious SMS Detection Using Ensemble Learning and SMOTE"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

from flask import Flask, request, jsonify
from flask_cors import CORS

# 1. Load dataset from CSV
df = pd.read_csv('dataset.csv')

# 2. Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply preprocessing
df["clean_text"] = df["text"].apply(clean_text)

# 3. Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])  # ham=0, smish=1

# 4. TF-IDF feature extraction
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df["clean_text"])

# 5. SMOTE oversampling to balance data
smote = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# 7. Define base models
svm = SVC(probability=True, kernel='linear', random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 8. Voting ensemble: SVM + RF
ensemble_model = VotingClassifier(estimators=[
    ('svm', svm),
    ('rf', rf)
], voting='soft')

# 9. Train ensemble model
ensemble_model.fit(X_train, y_train)

# 10. Evaluate model
y_pred = ensemble_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 11. Flask API
app = Flask(__name__)
CORS(app)  # Allow all domains for CORS

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sms_text = data.get('text', '')
    if not sms_text:
        return jsonify({"error": "No SMS text provided"}), 400

    cleaned = clean_text(sms_text)
    vector = vectorizer.transform([cleaned])
    pred = ensemble_model.predict(vector)
    prob = ensemble_model.predict_proba(vector)
    label = label_encoder.inverse_transform(pred)[0]

    report = {
        "input": sms_text,
        "prediction": label,
        "confidence": {
            "ham": round(float(prob[0][label_encoder.transform(['ham'])[0]]) * 100, 2),
            "smish": round(float(prob[0][label_encoder.transform(['smish'])[0]]) * 100, 2)
        }
    }
    return jsonify(report)

# To run the app (uncomment below if running standalone)
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
