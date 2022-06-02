from transformers import pipeline
from flask import Flask, request, jsonify, render_template

# initialize sentiment model
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# get sentiment
def sentiment_prediction(sentence):
    return classifier(sentence)

# initialize Flask app
app = Flask(__name__)

# creating endpoints
@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = str(request.form['sentence'])
    response = sentiment_prediction(sentence)[0]
    label = response['label']
    score = response['score']
    return render_template('index.html', prediction_text=f'Sentiment predicted is {label} with probability {score}')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 3000)