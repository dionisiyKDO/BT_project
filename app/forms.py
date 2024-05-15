from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, PasswordField
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.validators import InputRequired, Length, ValidationError
from flask_uploads import UploadSet, IMAGES
from app.models import User

images = UploadSet('images', IMAGES)

class UploadForm(FlaskForm):
    image = FileField(validators=[FileAllowed(images, 'Images only!'), FileRequired('File field should not be empty!')])
    submit = SubmitField('Upload')

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
