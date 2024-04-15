from neural_network import *

from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            # Save the uploaded file
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Classify
            checkpoint_path = './checkpoints/OwnV2.epoch36-val_acc0.9922.hdf5'
            loaded_model = load_model_from_checkpoint(checkpoint_path, 'OwnV2')
            predicted_class = classify_image(file_path, loaded_model)
            return render_template('index.html', text=predicted_class)
    return render_template('index.html')
  

if __name__ == "__main__":
    app.run(debug=True)
    