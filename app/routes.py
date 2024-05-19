from neural_network import MRIImageClassifier
from app import app

from app.forms import CommentForm, UploadForm, LoginForm, RegisterForm, mri_scans, BatchUploadForm
from app.models import db, User, Doctor, Patient, MRIScan, Comment 

from flask import flash, render_template, redirect, url_for, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_uploads import configure_uploads
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt

from datetime import datetime, timezone
import os

# Initialize extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
configure_uploads(app, mri_scans)

# Create database tables if they don't exist
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



@app.route('/', methods=['GET', 'POST'])
def index():
    return redirect(url_for('home'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('profile'))
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(
            email=form.email.data,
            username=form.username.data,
            password=hashed_password,
            role=form.role.data
        )
        db.session.add(user)
        db.session.commit()

        if form.role.data == 'doctor':
            doctor = Doctor(
                user_id=user.id,
                first_name=form.first_name.data,
                last_name=form.last_name.data,
                birth_date=form.birth_date.data,
                specialty=form.specialty.data
            )
            db.session.add(doctor)
        else:
            patient = Patient(
                user_id=user.id,
                first_name=form.first_name.data,
                last_name=form.last_name.data,
                birth_date=form.birth_date.data
            )
            db.session.add(patient)
        db.session.commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if current_user.role == 'doctor':
        mri_scans = MRIScan.query.filter_by(diagnosed_by=current_user.doctor.id).all()
        return render_template('profile_doctor.html', mri_scans=mri_scans, user=current_user)
    else:
        mri_scans = MRIScan.query.filter_by(patient_id=current_user.patient.id).all()
        return render_template('profile_patient.html', mri_scans=mri_scans, user=current_user)

@app.route('/profile/<filename>', methods=['GET', 'POST'])
@login_required
def image(filename):
    # return render_template('profile_patient.html', mri_scans=mri_scans, user=current_user)

    if not current_user.is_authenticated or current_user.role != 'doctor':
        flash("Access denied.", "danger")
        return redirect(url_for('index'))

    # Query the MRI scan by filename
    mri_scan = MRIScan.query.filter_by(file_name=filename).first()
    if not mri_scan:
        flash("MRI scan not found.", "danger")
        return redirect(url_for('profile'))

    form = CommentForm()
    if form.validate_on_submit():
        new_comment = Comment(
            mri_scan_id=mri_scan.id,
            doctor_id=current_user.doctor.id,
            comment=form.comment.data,
            created_at=datetime.now(timezone.utc)
        )
        db.session.add(new_comment)
        db.session.commit()
        flash("Comment added successfully.", "success")
        return redirect(url_for('image', filename=filename))

    # Query all comments related to the MRI scan
    comments = Comment.query.filter_by(mri_scan_id=mri_scan.id).all()

    return render_template('comment_form.html', mri_scan=mri_scan, comments=comments, form=form)




@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        file_name = mri_scans.save(form.image.data)
        patient = form.patient.data
        file_url = url_for('get_file', filename=file_name)
        predicted_class = model.classify_image(os.path.join(app.config['UPLOADED_IMAGES_DEST'], file_name))
        mri_scan = MRIScan(
            file_name=file_name,
            diagnosis=predicted_class,
            upload_date=datetime.now(timezone.utc),
            patient=patient,
            diagnosed_by_doctor=current_user.doctor,
        )
        db.session.add(mri_scan)
        db.session.commit()
        flash('Image uploaded and classified successfully!', 'success')
        return render_template('upload.html', form=form, text=predicted_class, file_url=file_url, user=current_user)
    return render_template('upload.html', form=form, user=current_user)

@app.route('/batch_upload', methods=['GET', 'POST'])
@login_required
def batch_upload():
    form = BatchUploadForm()
    if form.validate_on_submit():
        uploaded_files = form.images.data
        results = []
        for uploaded_file in uploaded_files:
            if uploaded_file:
                file_name = mri_scans.save(uploaded_file)
                predicted_class = model.classify_image(os.path.join(app.config['UPLOADED_IMAGES_DEST'], file_name))
                results.append((file_name, predicted_class))
        flash('Images uploaded and classified successfully!', 'success')
        return render_template('batch_upload.html', form=form, results=results, user=current_user)
    
    return render_template('batch_upload.html', form=form, user=current_user)