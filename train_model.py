import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load CSV
data = pd.read_csv('intents.csv')

# Drop empty rows
data = data.dropna(subset=['question'])
data = data.reset_index(drop=True)

# Features & labels
X = data['question'].astype(str)
y = data['intent']

# Vectorizer
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model trained successfully!")