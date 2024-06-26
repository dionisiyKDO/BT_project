from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, PasswordField, SelectField, DateField, TextAreaField, BooleanField
from flask_wtf.file import FileField, FileRequired, FileAllowed, MultipleFileField
from wtforms.validators import InputRequired, Length, ValidationError, Email, DataRequired, EqualTo
from flask_uploads import UploadSet, IMAGES
from app.models import User, Patient

mri_scans = UploadSet('images', IMAGES)

# Upload forms
# region
class UploadForm(FlaskForm):
    image       = FileField  ('Upload Image',   validators=[FileAllowed(mri_scans, 'Images only!'), FileRequired('File field should not be empty!')])
    search      = StringField('Search Patient', validators=[InputRequired()])
    patient_id  = StringField('Patient ID',     validators=[InputRequired()])
    submit      = SubmitField('Upload')

class BatchUploadForm(FlaskForm):
    images = MultipleFileField('Upload Images', validators=[FileAllowed(mri_scans, 'Images only!'), FileRequired('File field should not be empty!')])
    submit = SubmitField('Upload')
# endregion

# Search and Conclusion forms
# region
class SearchForm(FlaskForm):
    search      = StringField('Search Patient', validators=[InputRequired()])
    patient_id  = StringField('Patient ID',     validators=[InputRequired()])
    submit      = SubmitField('Search')

class ConclusionForm(FlaskForm):
    conclusion  = TextAreaField('Conclusion', validators=[InputRequired()], render_kw={"rows": 4})
    submit      = SubmitField  ('Submit')

class DiagnosisForm(FlaskForm):
    diagnosis   = StringField('Change diagnosis', validators=[DataRequired()])
    submit      = SubmitField('Upload')
# endregion

# Login and Registration forms
# region
class LoginForm(FlaskForm):
    email    = StringField  ('Email',    validators=[InputRequired()],                        render_kw={"placeholder": "Email"})
    password = PasswordField('Password', validators=[InputRequired(), Length(min=5, max=50)], render_kw={"placeholder": "Password"})
    submit   = SubmitField  ('Login')

class RegisterPatientForm(FlaskForm):
    email            = StringField  ('Email',            validators=[InputRequired(), Email()],               render_kw={"placeholder": "Email"})
    username         = StringField  ('Username',         validators=[InputRequired(), Length(min=4, max=30)], render_kw={"placeholder": "Username"})
    password         = PasswordField('Password',         validators=[InputRequired(), Length(min=8, max=50)], render_kw={"placeholder": "Password"})
    confirm_password = PasswordField('Confirm Password', validators=[InputRequired(), EqualTo('password')],   render_kw={"placeholder": "Confirm Password"})
    
    first_name = StringField('First Name', validators=[InputRequired(), Length(min=2, max=50)], render_kw={"placeholder": "First Name"})
    last_name  = StringField('Last Name',  validators=[InputRequired(), Length(min=2, max=50)], render_kw={"placeholder": "Last Name"})
    birth_date = DateField  ('Birth Date', validators=[InputRequired()], format='%Y-%m-%d')
    
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user: raise ValidationError('Username already taken!')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user: raise ValidationError('That email is already taken. Please choose a different one.')

class RegisterDoctorForm(FlaskForm):
    email            = StringField  ('Email',            validators=[InputRequired(), Email()],               render_kw={"placeholder": "Email"})
    username         = StringField  ('Username',         validators=[InputRequired(), Length(min=4, max=30)], render_kw={"placeholder": "Username"})
    password         = PasswordField('Password',         validators=[InputRequired(), Length(min=8, max=50)], render_kw={"placeholder": "Password"})
    confirm_password = PasswordField('Confirm Password', validators=[InputRequired(), EqualTo('password')],   render_kw={"placeholder": "Confirm Password"})
    
    first_name = StringField('First Name', validators=[InputRequired(), Length(min=2, max=50)], render_kw={"placeholder": "First Name"})
    last_name  = StringField('Last Name',  validators=[InputRequired(), Length(min=2, max=50)], render_kw={"placeholder": "Last Name"})
    birth_date = DateField  ('Birth Date', validators=[InputRequired()], format='%Y-%m-%d',     render_kw={"placeholder": "Birthday YYYY-MM-DD"})
    specialty  = StringField('Specialty', render_kw={"placeholder": "Specialy"})
    
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user: raise ValidationError('Username already taken!')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user: raise ValidationError('That email is already taken. Please choose a different one.')
# endregion

# Admin 
# region
class UserForm(FlaskForm):
    username    = StringField('Username',   validators=[DataRequired()])
    email       = StringField('Email',      validators=[DataRequired(), Email()])
    role        = SelectField('Role', choices=[('admin', 'Admin'), ('doctor', 'Doctor'), ('patient', 'Patient')], validators=[DataRequired()])
    first_name  = StringField('First name', validators=[DataRequired()])
    last_name   = StringField('Last name',  validators=[DataRequired()])
    is_active   = BooleanField('Active')
    submit      = SubmitField('Update')
    delete      = SubmitField('Delete')
# endregion