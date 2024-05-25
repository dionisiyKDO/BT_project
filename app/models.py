from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, timezone, date
import os

# to recreate the test.db file uncomment the following code 
# uncomment + create_databese_items() function in run.py
# if os.path.exists('./instance/test.db'):
#     os.remove('./instance/test.db')
#     print('test.db file deleted')
# else:
#     print('test.db does not exist')

db = SQLAlchemy()


class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    email      = db.Column(db.String(255), nullable=False, unique=True)
    username   = db.Column(db.String(100), nullable=False, unique=True)
    password   = db.Column(db.String(100), nullable=False)

    role       = db.Column(db.Enum('admin', 'doctor', 'patient', name='user_roles'), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc))
    is_active  = db.Column(db.Boolean, default=True)

    doctor     = db.relationship('Doctor',  back_populates='user', uselist=False)
    patient    = db.relationship('Patient', back_populates='user', uselist=False)


class Doctor(db.Model):
    __tablename__ = 'doctors'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    first_name = db.Column(db.String(255), nullable=False)
    last_name  = db.Column(db.String(255), nullable=False)
    birth_date = db.Column(db.Date, nullable=False)
    specialty  = db.Column(db.String(255))

    user       = db.relationship('User',    back_populates='doctor')
    mri_scans  = db.relationship('MRIScan', back_populates='diagnosed_by_doctor')
    conclusions   = db.relationship('Conclusion', back_populates='doctor')

class Patient(db.Model):
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    first_name = db.Column(db.String(255), nullable=False)
    last_name  = db.Column(db.String(255), nullable=False)
    birth_date = db.Column(db.Date, nullable=False)

    user       = db.relationship('User',    back_populates='patient')
    mri_scans  = db.relationship('MRIScan', back_populates='patient')

    def __str__(self) -> str:
        return f'{self.first_name} {self.last_name}'
    # def __repr__(self) -> str:
    #     return f'{self.first_name} {self.last_name}'

class MRIScan(db.Model):
    __tablename__ = 'mri_scans'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)

    file_name    = db.Column(db.String(255), nullable=False)
    upload_date  = db.Column(db.DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc))

    diagnosis    = db.Column(db.String(255), nullable=False)
    diagnosed_by = db.Column(db.Integer, db.ForeignKey('doctors.id'))

    conclusions            = db.relationship('Conclusion', back_populates='mri_scan')
    patient             = db.relationship('Patient', back_populates='mri_scans')
    diagnosed_by_doctor = db.relationship('Doctor',  back_populates='mri_scans')

class Conclusion(db.Model):
    __tablename__ = 'conclusions'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    mri_scan_id = db.Column(db.Integer, db.ForeignKey('mri_scans.id'), nullable=False)

    doctor_id  = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    conclusion = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc))

    mri_scan   = db.relationship('MRIScan', back_populates='conclusions')
    doctor     = db.relationship('Doctor',  back_populates='conclusions')
