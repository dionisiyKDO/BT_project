from app import app
from app.models import db, User, Doctor, Patient, MRIScan, Comment
from datetime import date, datetime, timezone
import os

def create_database_items():
    """
    Creates and initializes the database tables and adds sample data.

    Returns:
        None
    """
    with app.app_context():
        db.create_all()
        print('Database tables created')
        
        # Create users
        user1 = User(email='user1_patient@example.com', username='user1_patient', password='password1', role='patient')
        user2 = User(email='user2_doctor@example.com', username='user2_doctor', password='password2', role='doctor')
        user3 = User(email='user3_patient@example.com', username='user3_patient', password='password3', role='patient')
        user4 = User(email='user4_doctor@example.com', username='user4_doctor', password='password4', role='doctor')

        # Create doctors
        doctor1 = Doctor(user=user2, first_name='John', last_name='Doe', specialty='Neurology')
        doctor2 = Doctor(user=user4, first_name='Jane', last_name='Smith', specialty='Radiology')

        # Create patients
        patient1 = Patient(user=user1, first_name='Alice', last_name='Johnson', birth_date=date(1990, 1, 1))
        patient2 = Patient(user=user3, first_name='Bob', last_name='Williams', birth_date=date(1985, 5, 10))

        # Add items to the database session
        try:
            db.session.add_all([user1, user2, user3, user4, doctor1, doctor2, patient1, patient2])
            db.session.commit()
            print('Items created successfully')
        except Exception as e:
            db.session.rollback()
            print(f'Error creating items: {e}')

        # Create MRI scans
        scan1 = MRIScan(patient=patient1, file_name='scan1.jpg', upload_date=datetime.now(timezone.utc), diagnosis='glioma', diagnosed_by_doctor=doctor1)
        scan2 = MRIScan(patient=patient1, file_name='scan2.jpg', upload_date=datetime.now(timezone.utc), diagnosis='meningioma', diagnosed_by_doctor=doctor1)
        scan3 = MRIScan(patient=patient2, file_name='scan3.jpg', upload_date=datetime.now(timezone.utc), diagnosis='pituitary', diagnosed_by_doctor=doctor1)
        scan4 = MRIScan(patient=patient2, file_name='scan4.jpg', upload_date=datetime.now(timezone.utc), diagnosis='notumor', diagnosed_by_doctor=doctor2)

        # Create comments
        comment1 = Comment(mri_scan=scan1, doctor=doctor1, comment='This scan shows a tumor.')
        comment2 = Comment(mri_scan=scan3, doctor=doctor2, comment='No abnormalities found.')

        # Add items to the database session
        try:
            db.session.add_all([scan1, scan2, scan3, scan4, comment1, comment2])
            db.session.commit()
            print('Items created successfully')
        except Exception as e:
            db.session.rollback()
            print(f'Error creating items: {e}')

if __name__ == "__main__":
    # create_database_items()
    app.run(debug=True)