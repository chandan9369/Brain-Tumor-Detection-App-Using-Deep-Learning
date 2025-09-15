from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "hidden_key"

# Load the trained model
MODEL_PATH = 'models/model.keras'
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"⚠️ Error loading model: {e}")

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB


# ---------------- Helper Functions ----------------
def predict_tumor(image_path):
    """Load image, preprocess, and predict tumor type."""
    if model is None:
        return "Model not loaded", 0.0

    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = float(np.max(predictions, axis=1)[0])

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor Detected", confidence_score
    else:
        return f"Tumor Detected: {class_labels[predicted_class_index].capitalize()}", confidence_score


# ---------------- Routes ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("⚠️ No file part in request.", "danger")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("⚠️ No file selected. Please upload an MRI scan.", "warning")
            return redirect(request.url)

        if file:
            # Secure unique filename (avoid collisions)
            file_ext = os.path.splitext(file.filename)[1]
            filename = f"{uuid.uuid4().hex}{file_ext}"
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)

            # Predict tumor
            result, confidence = predict_tumor(file_location)

            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence * 100:.2f}",
                file_path=url_for('get_uploaded_file', filename=filename)
            )

    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    """Serve uploaded MRI scans."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ---------------- Run ----------------
if __name__ == '__main__':
    app.run(debug=True)
