# AI Chatbot (Scikit-Learn & Flask)

A lightweight intent-classification chatbot that uses Natural Language Processing (NLP) to understand user queries and respond based on predefined intents.

## 🚀 Features
- **Machine Learning**: Built with `Scikit-Learn` using a Logistic Regression model.
- **Text Vectorization**: Uses `TfidfVectorizer` to handle word importance.
- **Web API**: Integrated with `Flask` for easy communication with a frontend.

## 🛠️ Project Structure
- `train.py`: The script you provided to train the model and save `model.pkl` and `vectorizer.pkl`.
- `intents.csv`: Your dataset containing `question` and `intent` columns.
- `app.py`: The Flask server that loads the saved models and serves predictions.

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com
   cd ai-chatbot
