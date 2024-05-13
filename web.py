from neural_network import *

from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from wtforms import SubmitField, StringField, PasswordField
from wtforms.validators import DataRequired, Length, InputRequired, ValidationError
from flask_bcrypt import Bcrypt
import os

# Load the model
# region
checkpoint_path = './checkpoints/OwnV2.epoch36-val_acc0.9922.hdf5'
model = MRIImageClassifier()
model.load_model_from_checkpoint(checkpoint_path)
# endregion

app = Flask(__name__)

# Configuration
# region
UPLOAD_FOLDER = 'uploads'
SECRET_KEY = 'thisisasecretkey'
SQLALCHEMY_DATABASE_URI = 'sqlite:///test.db'

app.config['UPLOADED_IMAGES_DEST'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

images = UploadSet('images', IMAGES)
configure_uploads(app, images)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# endregion

# Upload page
# region

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
@login_required
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        filename = images.save(form.image.data)
        file_url = url_for('get_file', filename=filename)
        predicted_class = model.classify_image(os.path.join(app.config['UPLOADED_IMAGES_DEST'], filename))
        return render_template('upload.html', form=form, text=predicted_class, file_url=file_url, user=current_user)
    return render_template('upload.html', form=form, user=current_user)

# endregion


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=30)], render_kw={"placeholder": "Username"})
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=50)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=30)], render_kw={"placeholder": "Username"})
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=50)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user: raise ValidationError('Username already taken!')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))

    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

if __name__ == "__main__":
    # with app.app_context():
    #     db.create_all()
    app.run(debug=True)
    