import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("cifar10_model.h5")

# Load the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load an image you want to classify
img_path = "frog.jpg"  # Replace with the path to your image
img = image.load_img(img_path, target_size=(32, 32))  # Resize the image to match the model's input size
img_array = image.img_to_array(img)  # Convert the image to a Numpy array
img_array = img_array / 255.0  # Normalize the pixel values
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

# Make prediction
predictions = model.predict(img_array)

# Since the model returns logits, you might want to apply softmax to get probabilities
probabilities = tf.nn.softmax(predictions[0])

# Get the class with the highest probability
predicted_class = np.argmax(probabilities)

# Print the predicted class and its probability
print("Predicted class:", class_names[predicted_class])
print("Probability:", probabilities[predicted_class].numpy())
