import re
from pathlib import Path
import joblib
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#dapat parehas scikit version natin: 1.9.1

BASE = Path(__file__).parent
model = joblib.load(BASE / 'bow_model.joblib')
vectorizer = joblib.load(BASE / 'bow_vectorizer.joblib')

keep = {'full', 'down', 'up', 'top', 'bottom', 'front', 'back', 'side', 'system', 'empty'}
extra = {'doesn', 'don', 'isn', 'didn', 'wasn', 'aren', 'weren', 'couldn',
         'wouldn', 'shouldn', 'hasn', 'haven', 'hadn', 'won', 'll', 've', 're'}
stop = (set(ENGLISH_STOP_WORDS) - keep) | extra

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z]+', ' ', text)
    words = [w for w in text.split() if len(w) > 1 and w not in stop]
    return ' '.join(words)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    cleaned = clean(str(data.get('title', '')))
    if cleaned == '':
        return jsonify({'error': 'no usable words in the title'}), 400
    probs = model.predict_proba(vectorizer.transform([cleaned]))[0]
    percentages = {c: round(float(p) * 100, 2) for c, p in zip(model.classes_, probs)}
    category = max(percentages, key=percentages.get)
    return jsonify({'category': category, 'percentages': percentages})

if __name__ == '__main__':
    app.run(port=5000)

