import requests

url = 'http://127.0.0.1:5000/predict'

# Example input text to predict the category
input_data = {'text': 'This is an article about sports and football'}

response = requests.post(url, json=input_data)
print(response.json())  # Display the prediction result