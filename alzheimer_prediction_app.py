import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Function to preprocess the image
def preprocess_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict_alzheimer(image_path):
    input_size = (176, 176)  # Assuming this is the input size expected by your models
    
    # Preprocess the image
    image = preprocess_image(image_path, input_size)
    
    # Load the pre-trained models
    cnn_model = load_model('model/cnn5-3.h5')
    mobilenet_model = load_model('model/Mobilenet.h5')
    vgg_model = load_model('my_model.keras')
    
    # Predictions using CNN model
    cnn_prediction = cnn_model.predict(image)
    
    # Predictions using MobileNet model
    mobilenet_prediction = mobilenet_model.predict(image)
    
    # Predictions using VGG16 model
    vgg_prediction = vgg_model.predict(image)
    
    # Aggregate predictions (e.g., average probabilities)
    final_prediction = (cnn_prediction + mobilenet_prediction + vgg_prediction) / 3.0
    
    # Interpret final_prediction based on your model's output format
    # For simplicity, let's assume the highest probability corresponds to the predicted class
    predicted_class = np.argmax(final_prediction)
    
    return predicted_class

# Function to handle file selection
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        # Display the selected image
        img = Image.open(file_path)
        img = img.resize((300, 300))  # Resize the image for display
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Predict the class
        predicted_class = predict_alzheimer(file_path)
        messagebox.showinfo("Prediction Result", f"The predicted class is: {predicted_class}")

# Create a Tkinter window
root = tk.Tk()
root.title("Alzheimer's Disease Prediction")
root.geometry("400x400")

# Create a button to select an image file
select_button = tk.Button(root, text="Select Image", command=select_file)
select_button.pack(pady=10)

# Create a label to display the selected image
image_label = tk.Label(root)
image_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
