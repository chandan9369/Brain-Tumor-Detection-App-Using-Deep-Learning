from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# --------------------------
# Initialize Flask app
# --------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flashing messages

# --------------------------
# Configurations
# --------------------------
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --------------------------
# Load the trained model
# --------------------------
model = load_model('models/model.h5')

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# --------------------------
# Helper functions
# --------------------------
def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_tumor(image_path):
    """Predict tumor type from image."""
    try:
        IMAGE_SIZE = 128
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]

        if class_labels[predicted_class_index] == 'notumor':
            return "No Tumor", confidence_score
        else:
            return f"Tumor: {class_labels[predicted_class_index]}", confidence_score
    except Exception as e:
        return f"Error during prediction: {e}", 0.0

# --------------------------
# Routes
# --------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)
            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence*100:.2f}%",
                file_path=f'/uploads/{file.filename}'
            )
        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg, gif')
            return redirect(request.url)

    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --------------------------
# Run app
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
