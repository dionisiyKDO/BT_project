from neural_network import MRIImageClassifier
from app import app

from app.forms import CommentForm, UploadForm, LoginForm, RegisterForm, mri_scans, BatchUploadForm, SearchForm
from app.models import db, User, Doctor, Patient, MRIScan, Comment 

from flask import flash, jsonify, render_template, redirect, request, url_for, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_uploads import configure_uploads
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt

from datetime import datetime, timezone
import os
import pytz

# Initialize extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
configure_uploads(app, mri_scans)
utc_plus_3 = pytz.timezone('Etc/GMT-3')

# Create database tables if they don't exist
with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def get_file(filename):
    try:
        return send_from_directory(app.config['UPLOADED_IMAGES_DEST'], filename)
    except Exception as e:
        app.logger.error(f"Error sending file {filename}: {str(e)}")
        return redirect(url_for('home'))

@app.template_filter('formatdatetime')
def format_datetime(value, format='%d %B %Y %H:%M'):
    if value is None:
        return ""
    return value.strftime(format)

# Load the model
checkpoint_path = './checkpoints/OwnV2.epoch36-val_acc0.9922.hdf5'
model = MRIImageClassifier()
model.load_model_from_checkpoint(checkpoint_path)


# General routes
# region
@app.route('/', methods=['GET', 'POST'])
def index():
    return redirect(url_for('home'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        try:
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
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error during registration: {str(e)}")
            flash('Registration failed. Please try again.', 'danger')
    return render_template('register.html', form=form)

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

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))
# endregion

# Profile page
# region
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('q')
    patients = Patient.query.filter(
        db.or_(Patient.first_name.ilike(f'%{search}%'),
               Patient.last_name.ilike(f'%{search}%'))
    ).all()
    results = [{'id': patient.id, 'name': f'{patient.first_name} {patient.last_name}'} for patient in patients]
    return jsonify(matching_results=results)

@app.route('/mri_scans/<int:mri_scan_id>/comments', methods=['GET'])
def get_comments(mri_scan_id):
    mri_scan = MRIScan.query.get_or_404(mri_scan_id)
    comments = []
    for comment in mri_scan.comments:
        doctor = Doctor.query.filter_by(id=comment.doctor_id).first()
        comment_data = {
            'id': comment.id,
            'doctor': doctor.first_name + ' ' + doctor.last_name, 
            'comment': comment.comment,
            'created_at': comment.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        }
        comments.append(comment_data)
    return jsonify(comments)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = SearchForm()
    if current_user.role == 'doctor':
        if form.validate_on_submit():
            patient_id = form.patient_id.data
            mri_scans = MRIScan.query.filter_by(diagnosed_by=current_user.doctor.id, patient_id=patient_id).all()
        else:
            mri_scans = MRIScan.query.filter_by(diagnosed_by=current_user.doctor.id).all()
        return render_template('profile_doctor.html', mri_scans=mri_scans, user=current_user, form=form)
    else:
        mri_scans = MRIScan.query.filter_by(patient_id=current_user.patient.id).all()
        return render_template('profile_patient.html', mri_scans=mri_scans, user=current_user)

@app.route('/profile/<filename>', methods=['GET', 'POST'])
@login_required
def image(filename):
    if current_user.role == 'doctor':
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
            try:
                new_comment = Comment(
                    mri_scan_id=mri_scan.id,
                    doctor_id=current_user.doctor.id,
                    comment=form.comment.data,
                    created_at=datetime.now(utc_plus_3)
                )
                db.session.add(new_comment)
                db.session.commit()
                flash("Comment added successfully.", "success")
                return redirect(url_for('image', filename=filename))
            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Error adding comment: {str(e)}")
                flash("Failed to add comment. Please try again.", "danger")

        # Query all comments related to the MRI scan
        comments = Comment.query.filter_by(mri_scan_id=mri_scan.id).all()
        return render_template('comment_form.html', mri_scan=mri_scan, comments=comments, form=form)
    else:
        redirect(url_for('profile'))

@app.route('/delete_image', methods=['GET', 'POST'])
@login_required
def delete_image():
    image_id = request.args.get('id')
    mri_scan = MRIScan.query.filter_by(id=image_id).first()
    if not mri_scan:
        flash("MRI scan not found.", "danger")
        return redirect(url_for('profile'))
    try:
        file_path = os.path.join(app.config['UPLOADED_IMAGES_DEST'], mri_scan.file_name)
        os.remove(file_path)
        db.session.delete(mri_scan)
        db.session.commit()
        flash("Image deleted successfully.", "success")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting file {mri_scan.file_name}: {str(e)}")
        flash("Failed to delete image. Please try again.", "danger")
    return redirect(url_for('profile'))
# endregion

# Upload and classify MRI scans
# region
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        try:
            file_name = mri_scans.save(form.image.data)
            file_url = url_for('get_file', filename=file_name)
            patient = Patient.query.filter_by(id=form.patient_id.data).first()
            predicted_class = model.classify_image(os.path.join(app.config['UPLOADED_IMAGES_DEST'], file_name))
            mri_scan = MRIScan(
                file_name=file_name,
                diagnosis=predicted_class,
                upload_date=datetime.now(utc_plus_3),
                patient=patient,
                diagnosed_by_doctor=current_user.doctor,
            )
            db.session.add(mri_scan)
            db.session.commit()
            flash('Image uploaded and classified successfully!', 'success')
            return render_template('upload.html', form=form, text=predicted_class, file_url=file_url, user=current_user)
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error uploading image: {str(e)}")
            flash("Failed to upload image. Please try again.", "danger")
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
                try:
                    file_name = mri_scans.save(uploaded_file)
                    predicted_class = model.classify_image(os.path.join(app.config['UPLOADED_IMAGES_DEST'], file_name))
                    results.append((file_name, predicted_class))
                except Exception as e:
                    app.logger.error(f"Error during batch upload: {str(e)}")
                    flash("Failed to upload some images. Please try again.", "danger")
        flash('Images uploaded and classified successfully!', 'success')
        return render_template('batch_upload.html', form=form, results=results, user=current_user)
    return render_template('batch_upload.html', form=form, user=current_user)
# endregion