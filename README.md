# Brain MRI Analysis Platform

A web-based platform for MRI image analysis and patient management, combining medical expertise with machine learning for brain MRI classification. 
Built with Flask, TensorFlow, and secured user management system.

## Key Features

### Medical Professionals
- Image upload and management for patient MRI scans
- AI-powered image classification using CNN
- Comprehensive image viewing

### Patients
- View uploaded MRI images
- Access to medical conclusions and diagnoses
- Historical view of all examinations

### Administration
- Dedicated admin panel for system management
- Doctor registration and verification
- User management system
- Error logging and monitoring
- Neural network management:
  - Model retraining through web interface
  - Training progress visualization
  - Checkpoint selection and management

## Technical Stack

- **Backend**: Flask + Jinja2
- **Database**: SQLite using SQLAlchemy ORM
- **AI Model**: TensorFlow CNN for MRI classification
- **Security**: Role-based access control
- **Frontend**: PicoCSS

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
flask run
```

## Requirements

- Python 3.8+
- TensorFlow 2.10
- Sufficient storage for MRI images
- GPU recommended for model training

## Project Structure
```
BrainTumor/                      # Root directory
├── app/                         # Main application package
│   ├── config.py               # Application configuration
│   ├── forms.py                # WTForms definitions
│   ├── globals.py              # Global variables and constants
│   ├── models.py               # Database models
│   ├── routes.py               # URL route handlers
│   ├── static/                 # Static files directory
│   │   └── css/               # CSS stylesheets
│   │       ├── main.css       # Main styling
│   │       └── reset.css      # CSS reset
│   ├── templates/             # Jinja2 templates
│   │   ├── admin/            # Admin panel templates
│   │   │   ├── admin_profile.html      # Admin dashboard
│   │   │   ├── edit_user.html          # User editing interface
│   │   │   ├── errors.html             # Error logs view
│   │   │   ├── manage_users.html       # User management
│   │   │   ├── retrain.html            # NN retraining interface
│   │   │   └── select_checkpoint.html   # Model checkpoint selection
│   │   ├── base.html                   # Base template
│   │   ├── batch_upload.html           # Batch image upload
│   │   ├── conclusion_form.html        # Medical conclusion form
│   │   ├── error.html                  # Error page
│   │   ├── index.html                  # Landing page
│   │   ├── login.html                  # Login page
│   │   ├── profile_doctor.html         # Doctor's dashboard
│   │   ├── profile_patient.html        # Patient's dashboard
│   │   ├── register_doctor.html        # Doctor registration
│   │   ├── register_patient.html       # Patient registration
│   │   └── upload.html                 # Single image upload
│   └── uploads/                # Uploaded images storage
│
├── checkpoints/               # Neural network model checkpoints
│   ├── AlexNet.*.hdf5        # AlexNet model variants
│   └── OwnV*.*.hdf5          # Custom model variants
│
├── data/                     # Test/sample data directory
│   └── Te-*.jpg             # Test images
│
├── instance/                 # Instance-specific data
│   └── test.db              # SQLite database
│
├── MRIImageClassifier.py     # CNN model implementation
├── run.py                    # Application entry point
├── .gitignore               # Git ignore rules
└── .gitattributes           # Git attributes
```
