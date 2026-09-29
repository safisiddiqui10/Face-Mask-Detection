# 😷 Face Mask Detection

A deep learning-based **Face Mask Detection** system that uses a Convolutional Neural Network (CNN) to classify whether a person is wearing a face mask or not.

The project provides an interactive **Streamlit web interface** where users can provide an image and use the trained model for mask classification.

---

## 🚀 Features

* 😷 Detects whether a face is **With Mask** or **Without Mask**
* 🧠 CNN-based image classification
* ⚡ TensorFlow / Keras trained model
* 🖼️ Image-based prediction
* 🌐 Interactive Streamlit interface
* 📊 Displays prediction results
* 🧩 Modular and easy-to-extend project structure

---

## 🛠️ Tech Stack

| Technology | Purpose                    |
| ---------- | -------------------------- |
| Python     | Core programming language  |
| TensorFlow | Deep learning framework    |
| Keras      | Model building and loading |
| CNN        | Image classification       |
| NumPy      | Numerical operations       |
| Pillow     | Image processing           |
| Streamlit  | Web interface              |

---

## 🧠 Model

The project uses a **Convolutional Neural Network (CNN)** trained to classify facial images into two categories:

```text
with_mask
without_mask
```

The trained model is stored as a Keras `.h5` model.

Example:

```text
mask_nomaskmodelv2.h5
```

### Classification Flow

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
Resize Image
     │
     ▼
Normalize Pixel Values
     │
     ▼
CNN Model
     │
     ▼
Prediction
     │
     ├── With Mask
     │
     └── Without Mask
```

---

## 📂 Project Structure

```text
face-mask-detection/
│
├── app.py
├── mask_nomaskmodelv2.h5
├── requirements.txt
├── README.md
│
├── images/
│   └── sample images
│
└── .gitignore
```

> Your actual filenames may differ depending on your final project structure.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/face-mask-detection.git
```

Move into the project directory:

```bash
cd face-mask-detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 🖼️ How It Works

### Step 1 — Upload Image

Upload an image containing a person's face through the Streamlit interface.

### Step 2 — Preprocessing

The image is processed before being passed to the neural network.

Typical preprocessing includes:

```text
Image
  ↓
Resize
  ↓
Convert to Array
  ↓
Normalize Pixel Values
  ↓
Add Batch Dimension
```

### Step 3 — Model Prediction

The processed image is passed to the trained CNN model.

### Step 4 — Classification

The model predicts one of the two classes:

```text
WITH MASK
```

or

```text
WITHOUT MASK
```

---

## 📊 Model Input

The model expects an image resized to:

```text
128 × 128 pixels
```

Pixel values are normalized using:

```python
image = image / 255.0
```

The resulting image is then provided to the CNN model for prediction.

---

## 🧪 Example

```text
Input
  │
  ▼
┌──────────────────────┐
│   Person's Image     │
└──────────┬───────────┘
           │
           ▼
    Image Preprocessing
           │
           ▼
      CNN Model
           │
           ▼
   ┌───────┴────────┐
   │                │
   ▼                ▼
WITH MASK      WITHOUT MASK
```

---

## 🔮 Future Improvements

Potential improvements for future versions include:

* [ ] Real-time webcam detection
* [ ] Face detection before classification
* [ ] Multiple-face detection
* [ ] Confidence score display
* [ ] Bounding boxes around detected faces
* [ ] Improved CNN architecture
* [ ] Data augmentation
* [ ] Model performance evaluation
* [ ] Model deployment
* [ ] Docker support
* [ ] GPU acceleration

---

## 📈 Possible Future Architecture

The project can be extended from simple image classification into a complete real-time detection pipeline:

```text
Camera
   │
   ▼
Face Detection
   │
   ▼
Face Cropping
   │
   ▼
Image Preprocessing
   │
   ▼
CNN Classifier
   │
   ▼
Mask / No Mask
   │
   ▼
Confidence Score
```

---

## ⚠️ Limitations

This project is primarily designed as a machine-learning demonstration and may not perform perfectly in all real-world conditions.

Performance can vary depending on:

* Image quality
* Lighting conditions
* Face angle
* Occlusion
* Mask type
* Distance from camera
* Training dataset quality

---

## 📜 License

This project is available for educational and development purposes.

If you reuse or modify the project, please provide appropriate attribution.

---

## 👨‍💻 Author

**Your Name**

GitHub: `https://github.com/YOUR_USERNAME`

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.
