from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    
    # name    = db.Column(db.String(100), nullable=False)
    # surname = db.Column(db.String(100), nullable=False)

    images = db.relationship('Image')


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)

    # patient_name    = db.Column(db.String(100), nullable=False)
    # patient_surname = db.Column(db.String(100), nullable=False)
    # patient_age     = db.Column(db.Integer, nullable=False)

    diagnosis = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, default=datetime.now())
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
