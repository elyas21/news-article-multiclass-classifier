from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and the TfidfVectorizer
model = joblib.load('naive_bayes_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')  # If you saved the vectorizer too

# Define the home route to serve the HTML template
@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML template

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text data from the POST request
    data = request.get_json()
    text = data['text']
    
    # Preprocess the input text as you did before (tokenize, remove stopwords, etc.)
    processed_text = preprocess_text(text)
    
    # Transform the input text using the same vectorizer
    text_tfidf = tfidf.transform([processed_text])
    
    # Make prediction using the loaded model
    # Make prediction using the loaded model
    prediction = model.predict(text_tfidf)
    confidence = np.max(model.predict_proba(text_tfidf))  # Get the confidence score
    
    # Map class ID to class name
    class_names = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Science"
    }
    class_name = class_names.get(prediction[0], "Unknown")
    
    # Return the response in the desired format
    return jsonify({
        'status': 'success',
        'confidence': float(confidence),  # Convert numpy float to Python float
        'class': class_name
    })

# Function for preprocessing the input text (you can copy your preprocessing steps here)
def preprocess_text(text):
    # Here you should apply all the preprocessing steps like remove HTML, URLs, tokenization, stopwords removal, etc.
    # For this example, we are assuming preprocess_text returns the cleaned and preprocessed text
    return text  # Replace with actual preprocessing steps

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)