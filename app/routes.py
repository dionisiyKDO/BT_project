from neural_network import MRIImageClassifier
from app import app

from app.forms import *
from app.models import *

from flask import flash, jsonify, render_template, redirect, request, url_for, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_uploads import configure_uploads
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt

from datetime import datetime, timezone
import os
import pytz

# Configure the app
# region

# Initialize extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
configure_uploads(app, mri_scans)
utc_plus_3 = pytz.timezone('Etc/GMT-3')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def get_file(filename):
    try:
        return send_from_directory(app.config['UPLOADED_IMAGES_DEST'], filename)
    except Exception as e:
        app.logger.error(f"Error sending file {filename}: {str(e)}")
        flash('Failed to retrieve file.', 'danger')
        return redirect(url_for('home'))

@app.template_filter('formatdatetime')
def format_datetime(value, format='%d %B %Y %H:%M'):
    if value is None:
        return ""
    return value.strftime(format)

def create_admin_user():
    admin_email = 'admin'
    admin_username = 'admin'
    admin_password = generate_password_hash('admin')
    admin_role = 'admin'

    existing_admin = User.query.filter_by(email=admin_email).first()
    if not existing_admin:
        admin_user = User(
            email=admin_email,
            username=admin_username,
            password=admin_password,
            role=admin_role
        )
        db.session.add(admin_user)
        db.session.commit()
        app.logger.info("Admin user created.")

# Load the model
checkpoint_path = './checkpoints/OwnV2.epoch36-val_acc0.9922.hdf5'
model = MRIImageClassifier()
model.load_model_from_checkpoint(checkpoint_path)

# Create database tables if they don't exist
with app.app_context():
    db.create_all()
    create_admin_user()

# endregion

# General routes
# region
@app.route('/', methods=['GET', 'POST'])
def index():
    return redirect(url_for('home'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register_patient():
    form = RegisterPatientForm()
    if form.validate_on_submit():
        try:
            hashed_password = generate_password_hash(form.password.data)
            user = User(
                email=form.email.data,
                username=form.username.data,
                password=hashed_password,
                role='patient'
            )
            db.session.add(user)
            db.session.commit()
            patient = Patient(
                user_id=user.id,
                first_name=form.first_name.data,
                last_name=form.last_name.data,
                birth_date=form.birth_date.data
            )
            db.session.add(patient)
            db.session.commit()
            flash('Registration successful.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error during registration: {str(e)}")
            flash('Registration failed.', 'danger')
    return render_template('register_patient.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    errors = []
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            if user.role == 'admin':
                login_user(user)
                flash('Logged in as admin.', 'success')
                return redirect(url_for('admin_profile'))
            elif check_password_hash(user.password, form.password.data):
                if user.is_active:
                    login_user(user)
                    flash('Logged in successfully.', 'success')
                    return redirect(url_for('profile'))
                else:
                    errors.append('User is not active.')
                    flash('User is not active.', 'danger')
            else:
                errors.append('Invalid password.')
                flash('Invalid password.', 'danger')
        else:
            errors.append('User not found.')
            flash('User not found.', 'danger')
    return render_template('login.html', form=form, errors=errors)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
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

@app.route('/mri_scans/<int:mri_scan_id>/conclusions', methods=['GET'])
def get_conclusions(mri_scan_id):
    mri_scan = MRIScan.query.get_or_404(mri_scan_id)
    conclusions = []
    for conclusion in mri_scan.conclusions:
        doctor = Doctor.query.filter_by(id=conclusion.doctor_id).first()
        conclusion_data = {
            'id': conclusion.id,
            'doctor': doctor.first_name + ' ' + doctor.last_name, 
            'conclusion': conclusion.conclusion,
            'created_at': conclusion.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        }
        conclusions.append(conclusion_data)
    return jsonify(conclusions)

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

        mri_scan = MRIScan.query.filter_by(file_name=filename).first()
        if not mri_scan:
            flash("MRI scan not found.", "danger")
            return redirect(url_for('profile'))

        form = ConclusionForm()
        if form.validate_on_submit():
            try:
                new_conclusion = Conclusion(
                    mri_scan_id=mri_scan.id,
                    doctor_id=current_user.doctor.id,
                    conclusion=form.conclusion.data,
                    created_at=datetime.now(utc_plus_3)
                )
                db.session.add(new_conclusion)
                db.session.commit()
                flash("Conclusion added successfully.", "success")
                return redirect(url_for('image', filename=filename))
            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Error adding conclusion: {str(e)}")
                flash("Failed to add conclusion. Please try again.", "danger")

        conclusions = Conclusion.query.filter_by(mri_scan_id=mri_scan.id).all()
        return render_template('conclusion_form.html', mri_scan=mri_scan, conclusions=conclusions, form=form)
    else:
        flash('Access denied.', 'danger')
        return redirect(url_for('profile'))

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
        flash("Failed to delete image.", 'danger')
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
            flash("Failed to upload image.", "danger")
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

# admin
# region
@app.route('/admin')
@login_required
def admin_profile():
    if current_user.role != 'admin':
        flash('Access unauthorized!', 'danger')
        return redirect(url_for('index'))
    return render_template('admin/admin_profile.html')


@app.route('/admin/users')
@login_required
def manage_users():
    if current_user.role != 'admin':
        flash('Access unauthorized!', 'danger')
        return redirect(url_for('index'))
    users = User.query.all()
    return render_template('admin/manage_users.html', users=users)

@app.route('/admin/users/update/<int:user_id>', methods=['GET', 'POST'])
@login_required
def update_user(user_id):
    if current_user.role != 'admin':
        flash('Access unauthorized!', 'danger')
        return redirect(url_for('index'))
    user = User.query.get(user_id)
    form = UserForm(obj=user)
    if form.validate_on_submit():
        user.username = form.username.data
        user.email = form.email.data
        user.role = form.role.data
        user.is_active = form.is_active.data
        db.session.commit()
        flash('User updated successfully.', 'success')
        return redirect(url_for('manage_users'))
    return render_template('admin/edit_user.html', form=form, user=user)

@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        flash('Access unauthorized!', 'danger')
        return redirect(url_for('index'))
    
    user = User.query.get(user_id)
    if user:
        if user.role == 'doctor':
            doctor = Doctor.query.filter_by(user_id=user_id).first()
            if doctor:
                db.session.delete(doctor)
        elif user.role == 'patient':
            patient = Patient.query.filter_by(user_id=user_id).first()
            if patient:
                db.session.delete(patient)
                
        db.session.delete(user)
        db.session.commit()
        flash('User and related records deleted successfully.', 'success')
    else:
        flash('User not found.', 'danger')
    return redirect(url_for('manage_users'))

@app.route('/admin/register_doctor', methods=['GET', 'POST'])
@login_required
def register_doctor():
    if current_user.role != 'admin':
        flash('Access unauthorized!', 'danger')
        return redirect(url_for('index'))

    form = RegisterDoctorForm()
    if form.validate_on_submit():
        try:
            hashed_password = generate_password_hash(form.password.data)
            user = User(
                email=form.email.data,
                username=form.username.data,
                password=hashed_password,
                role='doctor'
            )
            db.session.add(user)
            db.session.commit()
            doctor = Doctor(
                user_id=user.id,
                first_name=form.first_name.data,
                last_name=form.last_name.data,
                specialty=form.specialty.data,
                birth_date=form.birth_date.data
            )
            db.session.add(doctor)
            db.session.commit()
            flash('Doctor registered successfully', 'success')
            return redirect(url_for('admin_profile'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error during registration: {str(e)}")
            flash('Registration failed. Please try again.', 'danger')
    return render_template('register_doctor.html', form=form)


@app.errorhandler(Exception)
def handle_error(error):
    error_log = ErrorLog(message=str(error))
    db.session.add(error_log)
    db.session.commit()
    return render_template('error.html', error=error_log), 500

@app.route('/trigger_error')
def trigger_error():
    raise Exception("This is a test error")

@app.route('/admin/errors')
@login_required
def view_errors():
    if current_user.role != 'admin':
        flash('Access unauthorized!', 'danger')
        return redirect(url_for('index'))
    errors = ErrorLog.query.order_by(ErrorLog.id.desc()).all()
    return render_template('admin/errors.html', errors=errors)
# endregion