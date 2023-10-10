import os
import numpy as np
import tensorflow.compat.v2 as tf
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Configuration
class_labels = ['Cercospora leaf spot (Gray leaf spot)', 'Common rust', 'Northern Leaf Blight', 'healthy']
class_preventive_measures = [
    '...',
    '...',
    '...',
    'Your plant is healthy :)'
]

img_rows, img_cols = 224, 224
image_size = [244, 244, 3]

# Load the model
def load_saved_model(model_path):
    global model
    model = load_model(model_path)
    print(" * Model loaded!")

# Configuration settings
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
MODEL_PATH = 'my_model.h5'
UPLOAD_FOLDER = 'static/images'

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to perform image prediction
def predict_disease(path):
    img = load_img(path, target_size=image_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32') / 255
    z = model.predict(x)
    index = np.argmax(z)
    accuracy = int(np.array(z).max() * 100)
    return [index, accuracy]

# Route for the homepage
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

# Route for image prediction
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            try:
                filename = file.filename
                file_extension = filename.split('.')[-1]
                file_path = os.path.join(UPLOAD_FOLDER, "testing-image." + file_extension)
                file.save(file_path)
                result = predict_disease(file_path)
                disease_name = class_labels[result[0]]
                accuracy = result[1]

                return render_template('predict.html',
                                       disease_name=disease_name,
                                       user_image=file_path,
                                       accuracy=accuracy,
                                       preventive_measures=class_preventive_measures[result[0]])
            except Exception as e:
                return "Error: " + str(e)
        else:
            eMessage = "Please upload a valid image file with extension: " + ", ".join(ALLOWED_EXTENSIONS)
            return redirect(url_for('predict', error=eMessage))
    elif request.method == 'GET':
        return redirect(url_for('predict'))

# Route to download images
@app.route('/download-image/<path:filename>')
def download(filename):
    return send_from_directory('static', filename, as_attachment=True, mimetype='image/jpg',
                               attachment_filename=(str(filename) + '.jpg'))

if __name__ == "__main__":
    load_saved_model(MODEL_PATH)
    app.run(debug=True)
