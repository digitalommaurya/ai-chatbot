from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load CSV responses
data = pd.read_csv('intents.csv')

@app.route('/')
def home():
    return render_template('index.html')  # renamed template

@app.route('/get', methods=['POST'])
def chatbot():
    user_input = request.form['message']

    # Transform input
    input_vec = vectorizer.transform([user_input])

    # Predict intent
    intent = model.predict(input_vec)[0]

    # Get response
    responses = data[data['intent'] == intent]['response'].values

    if len(responses) > 0:
        response = responses[0]
    else:
        response = "Sorry, I didn't understand. Can you rephrase?"

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)