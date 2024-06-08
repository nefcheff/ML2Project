# You can run this script and acces the appilation on http://127.0.0.1:5000
# Maybe you will need to adjust the paths.
# if you want to use another model safe model in Notebook MyCnnModel and adjust path for .keras file below.
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf


app = Flask(__name__, template_folder='../frontend/')

import tensorflow as tf

# Set path to model
model_path = os.path.abspath('../ML2Project/modelLib/model_4.keras') #path to model (.keras file), may need to be adjusted
print("Absolute path to model:", model_path)
model = tf.keras.models.load_model(model_path)

# Set path to class name
class_names_path = os.path.abspath('../ML2Project/modelLib/class_names.txt') #path to class names, may need to be adjusted
with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

# Define the image size
IMAGE_SIZE = (224, 224)

# Preprocessing function
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        img_path = os.path.join("uploads", file.filename)
        file.save(img_path)
        
        # Pre-process image and make prediction
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array)
        
        # Return probabilities for each class
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class]
        predictions_with_names = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        return jsonify({
            "predictions": predictions_with_names,
            "predicted_class": predicted_class_name
        })


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)