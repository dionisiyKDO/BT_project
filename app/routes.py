from app import app

from app.forms import UploadForm, LoginForm, RegisterForm, images
from app.models import db, User, Image

from neural_network import MRIImageClassifier

from flask import render_template, request, redirect, url_for, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_uploads import UploadSet, IMAGES, configure_uploads
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt
import os


db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
configure_uploads(app, images)

with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_IMAGES_DEST'], filename)
    
@app.template_filter('formatdatetime')
def format_datetime(value, format='%d %B %Y %H:%M'):
    """Format a date time to (HH:MM dd BBBB YYYY) format."""
    if value is None:
        return ""
    return value.strftime(format)

# Load the model
checkpoint_path = './checkpoints/OwnV2.epoch36-val_acc0.9922.hdf5'
model = MRIImageClassifier()
model.load_model_from_checkpoint(checkpoint_path)




@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        filename = images.save(form.image.data)
        file_url = url_for('get_file', filename=filename)
        print(f"File URL: {file_url}")  # Debugging line
        predicted_class = model.classify_image(os.path.join(app.config['UPLOADED_IMAGES_DEST'], filename))
        image = Image(filename=filename, diagnosis=predicted_class, doctor_id=current_user.id)
        db.session.add(image)
        db.session.commit()
        return render_template('upload.html', form=form, text=predicted_class, file_url=file_url, user=current_user)
    return render_template('upload.html', form=form, user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=True)
            return redirect(url_for('profile'))
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(username=form.username.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('home'))
    return render_template('register.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    images = Image.query.filter_by(doctor_id=current_user.id).all()
    return render_template('profile.html', images=images, user=current_user)

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')
