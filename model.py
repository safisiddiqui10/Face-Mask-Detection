import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Mask Detection App",
    layout="centered"
)

st.title("Mask Detection App")
st.write("Choose an image from your device or take a photo to detect mask.")

# Load model (cached)
@st.cache_resource
def load_model_cached():
    return load_model("MaskPrediction.keras")

model = load_model_cached()

# Class names
class_names = ['Mask', 'No Mask']

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Choose input method
input_option = st.radio("Select input method:", ("Upload Image", "Take Photo"))

image = None

if input_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif input_option == "Take Photo":
    captured_image = st.camera_input("Take a photo")
    if captured_image:
        image = Image.open(captured_image).convert("RGB")

# If an image is available, predict
if image:
    st.image(image, caption="Selected Image", width=200)
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)[0][0]

    # Determine label
    label = 1 if prediction >= 0.5 else 0

    # Display result
    if label == 1:
        st.error(f"Prediction: {class_names[label]}")
    else:
        st.success(f"Prediction: {class_names[label]}")
