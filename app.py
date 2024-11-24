from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import base64
import cv2

app = Flask(__name__)

# Load the trained model for face emotion detection
model = tf.keras.models.load_model('face_emotion_detection_model.h5')

# Load class names from a separate file or define them dynamically
class_names_file = 'class_names.txt'

if os.path.exists(class_names_file):
    with open(class_names_file, 'r') as f:
        emotion_classes = f.read().splitlines()
else:
    # Define emotion classes manually if class_names.txt does not exist
    emotion_classes = ['Class1', 'Class2', 'Class3']  # Modify as needed

# Define a function to preprocess image and make predictions
def predict_output(image_data):
    try:
        # Convert base64 encoded image data to numpy array
        decoded_image = base64.b64decode(image_data)
        nparr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (OpenCV uses BGR by default)
        img = cv2.resize(img, (64, 64))  # Resize image to match model input shape
        img = img.astype(np.float32) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        result = model.predict(img)
        predicted_class_index = np.argmax(result)
        predicted_class = emotion_classes[predicted_class_index]
        return predicted_class
    except Exception as e:
        print("Error during prediction:", str(e))
        return "Error"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected image'})

    try:
        image_data = image_file.read()
        image_data_b64 = base64.b64encode(image_data).decode('utf-8')
        predicted_emotion = predict_output(image_data_b64)
        return jsonify({'prediction': predicted_emotion})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
