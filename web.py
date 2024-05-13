from neural_network import *

from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os

# Load the model
checkpoint_path = './checkpoints/OwnV2.epoch36-val_acc0.9922.hdf5'
model = MRIImageClassifier()
model.load_model_from_checkpoint(checkpoint_path)

app = Flask(__name__)

# Configuration
# region
UPLOAD_FOLDER = 'uploads'
SECRET_KEY = 'thisisasecretkey'
SQLALCHEMY_DATABASE_URI = 'sqlite:///test.db'

app.config['UPLOADED_IMAGES_DEST'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = SECRET_KEY
# app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
# endregion

images = UploadSet('images', IMAGES)
configure_uploads(app, images)

class UploadForm(FlaskForm):
    image = FileField(
        validators=[
                FileAllowed(images, 'Images only!'),
                FileRequired('File field should not be empty!'),
            ]
        )
    submit = SubmitField('Upload')

@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_IMAGES_DEST'], filename)
    


@app.route('/upload', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        filename = images.save(form.image.data)
        file_url = url_for('get_file', filename=filename)
        predicted_class = model.classify_image(os.path.join(app.config['UPLOADED_IMAGES_DEST'], filename))
        return render_template('upload.html', form=form, text=predicted_class, file_url=file_url)
    return render_template('upload.html', form=form)
  
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
    