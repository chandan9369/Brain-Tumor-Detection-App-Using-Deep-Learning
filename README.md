# MRI Tumor Detection Web Application

This repository contains a Flask-based web application for detecting and classifying brain tumors from MRI scans using a convolutional neural network (CNN) built on top of the VGG16 pre-trained model. The application allows users to upload MRI images and receive predictions on the presence and type of tumor (pituitary, glioma, meningioma, or no tumor) along with a confidence score.

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


## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/mri-tumor-detection.git
   cd mri-tumor-detection
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include:
   - `flask`
   - `tensorflow`
   - `numpy`
   - `pillow` (for image processing)

4. **Download the Pre-trained Model**:
   - Place the trained model file (`model.h5`) in the `models/` directory. You can train your own model or download a pre-trained one (not included in this repository due to size).

5. **Create Uploads Directory**:
   ```bash
   mkdir uploads
   ```

## Usage
1. **Run the Flask Application**:
   ```bash
   python app.py
   ```
   The application will start in debug mode and be accessible at `http://127.0.0.1:5000`.

2. **Access the Web Interface**:
   - Open a web browser and navigate to `http://127.0.0.1:5000`.
   - Upload an MRI image using the provided interface.
   - View the prediction result and confidence score.

## File Structure
```
mri-tumor-detection/
│
├── app.py                  # Flask application script
├── index.html              # HTML template for the web interface
├── models/                 # Directory for the pre-trained model (model.h5)
├── uploads/                # Directory for uploaded MRI images
├── model_architecture.md   # Detailed model architecture documentation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Dependencies
- Python 3.8+
- Flask
- TensorFlow
- NumPy
- Pillow
