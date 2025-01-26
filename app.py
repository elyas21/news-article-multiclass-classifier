from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and the TfidfVectorizer
model = joblib.load('naive_bayes_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Serve the frontend
@app.route('/')
def home():
    return render_template('index.html')

# Serverless function for the /predict endpoint
def predict_handler(request):
    try:
        # Get the text data from the POST request
        data = request.get_json()
        text = data['text']
        
        # Preprocess the input text
        processed_text = preprocess_text(text)
        
        # Transform the input text using the same vectorizer
        text_tfidf = tfidf.transform([processed_text])
        
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
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'confidence': float(confidence),
                'class': class_name
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'message': str(e)
            })
        }

# Function for preprocessing the input text
def preprocess_text(text):
    # Add your preprocessing steps here
    return text

# Run the Flask app (for local development)
if __name__ == '__main__':
    app.run(debug=True)