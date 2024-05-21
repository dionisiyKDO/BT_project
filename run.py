from app import app
from app.models import db, User, Doctor, Patient, MRIScan, Comment
from datetime import date, datetime, timezone

if __name__ == "__main__":
    app.run(debug=True)