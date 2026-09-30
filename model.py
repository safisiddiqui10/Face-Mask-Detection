import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Mask Detection App",
    page_icon="😷",
    layout="centered"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("😷 Mask Detection App")
st.caption("Upload an image or use your camera to check whether a person is wearing a mask.")

st.divider()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model_cached():
    return load_model("MaskPrediction.keras")

model = load_model_cached()

# Class names
class_names = ["Mask", "No Mask"]


# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --------------------------------------------------
# INPUT METHOD
# --------------------------------------------------
st.subheader("📷 Choose Input")

input_option = st.radio(
    "Select how you want to provide an image:",
    ["Upload Image", "Take Photo"],
    horizontal=True
)

image = None


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------
if input_option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")


# --------------------------------------------------
# CAMERA
# --------------------------------------------------
else:

    captured_image = st.camera_input(
        "Take a photo"
    )

    if captured_image:
        image = Image.open(captured_image).convert("RGB")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if image:

    st.divider()

    st.subheader("🖼️ Image Preview")

    # Center-ish image using columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            image,
            caption="Selected Image",
            use_container_width=True
        )

    # Predict button
    if st.button(
        "🔍 Detect Mask",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            preprocessed_img = preprocess_image(image)

            prediction = model.predict(
                preprocessed_img,
                verbose=0
            )[0][0]

        # Determine label
        label = 1 if prediction >= 0.5 else 0

        # Confidence
        confidence = prediction if label == 1 else 1 - prediction
        confidence_percent = confidence * 100

        st.divider()

        st.subheader("📊 Detection Result")

        # Result
        if label == 0:

            st.success(
                f"### 😷 {class_names[label]}"
            )

            st.write(
                f"Confidence: **{confidence_percent:.2f}%**"
            )

        else:

            st.error(
                f"### 🚫 {class_names[label]}"
            )

            st.write(
                f"Confidence: **{confidence_percent:.2f}%**"
            )

        # Confidence bar
        st.progress(
            float(confidence),
            text=f"Confidence: {confidence_percent:.2f}%"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "😷 Face Mask Detection • Powered by TensorFlow & Streamlit"
)