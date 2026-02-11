import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------
# Load Model
# ------------------------------
model = tf.keras.models.load_model("tomato_disease_model.h5")

# Class Labels (Make sure order matches training)
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
st.caption("Developed by Vivek B")

uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize image
    image = image.resize((128,128))

    # Convert to numpy
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction Result")
    st.success(f"Disease: {class_labels[class_index]}")
    st.info(f"Confidence: {round(float(confidence)*100,2)}%")
