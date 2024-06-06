from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model(r"C:\Users\sandeepk\Desktop\alzheimer\cnn5-3.h5")

# Define class labels based on your training data
class_labels = {0: "MildDemented", 1: "ModerateDemented", 2: "NonDemented", 3: "VeryMildDemented"}

# Function to preprocess an image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(176, 176))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array

# Function to make predictions on a single image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    return predicted_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded image file
        file = request.files['file']
        if file:
            # Save the file temporarily
            file_path = 'temp.jpg'
            file.save(file_path)
            # Make prediction
            predicted_label = predict_image(file_path)
            return render_template('index.html', prediction=predicted_label)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
