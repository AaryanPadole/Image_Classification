import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Load the trained model
model = load_model("cifar10_model.h5")

# Load the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# URL of the image you want to classify
img_url = "https://etimg.etb2bimg.com/photo/103642063.cms"  # Replace with the URL of your image

# Download the image from the URL
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))

# Resize the image to match the model's input size
img = img.resize((32, 32))

# Convert the image to a Numpy array
img_array = image.img_to_array(img)

# Normalize the pixel values
img_array = img_array / 255.0

# Add a batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)

# Since the model returns logits, you might want to apply softmax to get probabilities
probabilities = tf.nn.softmax(predictions[0])

# Get the class with the highest probability
predicted_class = np.argmax(probabilities)

# Print the predicted class and its probability
print("Predicted class:", class_names[predicted_class])
print("Probability:", probabilities[predicted_class].numpy())
