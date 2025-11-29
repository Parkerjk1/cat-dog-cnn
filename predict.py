import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("cat_dog_model.keras")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0   # normalize
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "Dog"
    else:
        return "Cat"

# Example run
if __name__ == "__main__":
    img_path = input("Enter image file path: ")
    result = predict_image(img_path)
    print("Prediction:", result)

