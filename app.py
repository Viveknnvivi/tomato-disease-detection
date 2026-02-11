import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ------------------------------
# Load Model
# ------------------------------
model = tf.keras.models.load_model("tomato_disease_model.h5")

# Class Labels (must match training order)
class_labels = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Tomato Disease Detection")

st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to array
    img = np.array(image)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction Result")
    st.success(f"Disease: {class_labels[class_index]}")
    st.info(f"Confidence: {round(float(confidence)*100,2)} %")
    

st.caption("Developed by Vivek B")

