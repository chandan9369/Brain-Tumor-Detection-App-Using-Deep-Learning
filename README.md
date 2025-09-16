# 🧠 MRI Tumor Detection System

A Flask-based web application that detects **brain tumors from MRI scans** using a pre-trained deep learning model (`model.keras`).  

Users can upload an MRI image, and the system will:
- Preprocess the image  
- Run inference with a TensorFlow/Keras CNN model  
- Display the **tumor type (Glioma, Meningioma, Pituitary)** or **No Tumor**  
- Show the **confidence score** along with the uploaded image

## Features
- **Image Upload**: Users can upload MRI images in PNG, JPG, JPEG, or GIF formats.
- **Tumor Classification**: The model classifies the MRI scan into one of four categories: pituitary, glioma, meningioma, or no tumor.
- **Confidence Score**: Displays the confidence level of the prediction.
- **Web Interface**: A simple and intuitive interface built with HTML and Flask templates.
- **Pre-trained Model**: Leverages transfer learning with VGG16 pre-trained on ImageNet, fine-tuned for MRI tumor detection.

## Model Architecture
The model is built using the VGG16 architecture with the following modifications:
- **Base Model**: VGG16 with `input_shape=(128, 128, 3)`, `include_top=False`, and `weights='imagenet'`.
- **Frozen Layers**: Most VGG16 layers are frozen to retain pre-trained weights, with the last three layers set to trainable for fine-tuning.
- **Additional Layers**:
  - Flatten layer to convert 3D tensors to 1D.
  - Dropout layer (0.3) to prevent overfitting.
  - Dense layer with 128 neurons and ReLU activation.
  - Dropout layer (0.2) for additional regularization.
  - Output dense layer with 4 neurons (one for each class) and softmax activation.
- **Input**: MRI images resized to 128x128 pixels.
- **Output**: Probability distribution over four classes (pituitary, glioma, meningioma, notumor).

---

## 📸 Demo

![App Screenshot](templates/demo.png) <!-- optional: add a screenshot -->

---

## 🚀 Features
- Upload MRI images (`.jpg`, `.png`, etc.)  
- Automatic classification into:
  - **Glioma**
  - **Meningioma**
  - **Pituitary tumor**
  - **No Tumor**  
- Shows prediction confidence (%)  
- Clean Bootstrap UI with loading spinner & alerts  

---

## 🗂️ Project Structure

```
brain-tumor-detection/
│
├── app.py                 # Flask backend
├── models/
│   └── model.keras        # Trained deep learning model
├── uploads/               # Stores uploaded MRI scans (auto-created)
├── templates/
│   └── index.html         # Frontend (Bootstrap UI)
├── requirements.txt       # Dependencies
└── Procfile               # Startup config for Render
```

---

## ⚙️ Installation (Local Setup)

1. **Clone the repository**
   ```bash
   git clone https://github.com/chandan9369/Brain-Tumor-Detection-App-Using-Deep-Learning.git
   cd Brain-Tumor-Detection-App-Using-Deep-Learning
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**
   ```bash
   python app.py
   ```

5. Open your browser at:
   ```
   http://127.0.0.1:5000
   ```

---

## 🛠️ Tech Stack
- **Frontend:** HTML, Bootstrap 5  
- **Backend:** Flask (Python)  
- **Model:** TensorFlow/Keras CNN  

---

## 📊 Model Info
- Input size: **128x128 RGB**  
- Output classes: `['pituitary', 'glioma', 'notumor', 'meningioma']`  
- Saved format: `.keras`  

---

## ✨ Future Improvements
- Add MRI preview before uploading  
- Deploy with GPU support  
- Optimize model (reduce size, faster inference)  

