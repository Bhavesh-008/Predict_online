from flask import Flask, render_template, request
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load the encoder object from the file
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Define the function for padding
def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

# Define the function for prediction
def sample_predict(sample_pred_text, pad):
    # Encode the input text using the loaded encoder
    encoded_sample_pred_text = encoder.encode(sample_pred_text)

    # Perform padding if required
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)

    # Load the model and make predictions
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    # Determine the output based on the prediction
    output = ""
    if predictions[0][0] >= 0.5:
        output = "POSITIVE"
    elif predictions[0][0] <= -1:
        output = "NEGATIVE"
    else:
        output = "NEUTRAL"

    return output 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    pad = request.form.get('pad') == 'on'  # Checkbox value
    prediction = sample_predict(text, pad)
    return render_template('index.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)
