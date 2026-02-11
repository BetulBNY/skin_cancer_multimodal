
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import pickle
import pandas as pd

# Adding the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)  # Purpose: Adding the root of my project to Python's module search path so I can import local packages like preprocessing

"""
skin_cancer_project/          ← Bu seviyeyi sys.path'e ekliyoruz
├── preprocessing/            ← Bu modülü import edebilmek için
│   └──__init__.py
└── backend/                  ← app.py burada
    └── app.py
"""

# Loading Preprocessing Functions
try:
    #sys.path.append(r"C:\Users\betus\Desktop\all_projects\skin_cancer\skin_cancer_project\sprep")
    from preprocessing import create_age_group_column, apply_loc_mean_age, age_dev_from_loc_mean

    print("Preprocessing functions loaded successfully")
except ImportError as e:
    print(f"Failed to load preprocessing functions: {e}")


# Start Flask App
app = Flask(__name__)
CORS(app)  # Allows all domains for development (Cross-Origin Resource Sharing)

# A folder which uploaded files/images will be saved
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global Variables
model = None
preprocessor = None
mean_age_map = None

def load_model_and_preprocessor():
    """ Upload Model and preprocessor """
    global model, preprocessor, mean_age_map

    try:
        # Model upload
        model_path = os.path.join(os.path.dirname(__file__), 'final_multimodal_model.keras')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Model uploaded successfully")
        else:
            print(f"Model file not found: {model_path}")
            return False

        # Preprocessor upload
        preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            print("Preprocessor uploaded successfully")
        else:
            print(f"Preprocessor file not found: {preprocessor_path}")
            return False

        # Sample mean_age_map
        mean_age_map =  {
         'abdomen': 50.18283385965737, 'acral': 37.0, 'back': 55.46498054474708, 'chest': 57.18164991850531,
         'ear': 59.06976744186046, 'face': 63.38748121915441, 'foot': 47.855319137764226, 'genital': 47.15909090909091,
         'hand': 53.858695652173914, 'lower extremity': 53.75116495806151, 'neck': 57.29545454545455,
         'scalp': 64.42307692307692, 'trunk': 50.2066151173732, 'unknown': 49.290899713067176,
         'upper extremity': 57.64861007295935
                        }
        return True

    except Exception as e:
        print(f"Model or Preprocessor failed to upload: {e}")
        return False

def preprocess_image(image_bytes):
    """image preprocessing method"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print(f"Image processing error: {e}")
        return None

def prepare_tabular_data(age, sex, localization):
    """Prepare tabular data"""
    try:
        # Create DataFrame
        df = pd.DataFrame([{
            'age': float(age),
            'sex': sex,
            'localization': localization
        }])

        # Add age group
        df = create_age_group_column(df)

        # Add Loc_mean_age
        df['loc_mean_age'] = df['localization'].map(mean_age_map).fillna(50.0)

        # Add age deviation
        df['age_dev_from_loc_mean'] = df['age'] - df['loc_mean_age']

        # Just select necessary columns
        feature_columns = ['age', 'loc_mean_age', 'age_dev_from_loc_mean', 'sex', 'localization', 'age_group']
        df_features = df[feature_columns]

        # Transform with Preprocessor
        processed_data = preprocessor.transform(df_features)

        return processed_data

    except Exception as e:
        print(f"Tabular data preparation error: {e}")
        return None

# Check System Status
@app.route('/')
def home():
    """Home page"""
    return jsonify({
        'message': 'Skin Cancer Detection API is working huhu!',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/health')
def health_check():
    """Health Check"""
    return jsonify({
        'status': 'healthy',
        'model': 'loaded' if model is not None else 'not_loaded',
        'preprocessor': 'loaded' if preprocessor is not None else 'not_loaded'
    })

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    """Prediciton endpoint"""
    try:
        # Checking model
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model or preprocessor not loaded'}), 500

        # Image control
        if 'image' not in request.files:
            return jsonify({'error': 'Image not loaded'}), 400

        # Process image
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'invalid file'}), 400

        image_bytes = image_file.read()
        image_input = preprocess_image(image_bytes)

        if image_input is None:
            return jsonify({'error': 'Image could not be processed'}), 400

        # Retrieve form data and set default values
        age = float(request.form.get('age', 50))
        sex = request.form.get('sex', 'male')
        localization = request.form.get('localization', 'unknown')

        # Validation
        if age < 0 or age > 120:
            return jsonify({'error': 'Invalid age value'}), 400

        if sex not in ['male', 'female']:
            return jsonify({'error': 'Gender must be male or female'}), 400

        # Prepare tabular data
        tabular_input = prepare_tabular_data(age, sex, localization)

        if tabular_input is None:
            return jsonify({'error': 'Tabular data is not prepared'}), 400

        # Model predict
        prediction_prob = model.predict([image_input, tabular_input], verbose=0)[0][0]

        # Evaluate result
        result = "Malignant (Cancerous)" if prediction_prob > 0.5 else "Benign (Healthy)"
        confidence = float(prediction_prob)

        # Risk Level
        if confidence > 0.8:
            risk_level = "High"
        elif confidence > 0.6:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'risk_level': risk_level,
            'recommendation': 'Consult a doctor' if confidence > 0.7 else 'Follow'
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction Error: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint is not found :('}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Server error :('}), 500


if __name__ == '__main__':
    print(" Flask uygulaması başlatılıyor..")

    # Model ve preprocessor'u yükle
    if load_model_and_preprocessor():
        print("All components are installed, the application is ready!")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Critical components failed to load, application cannot start!")
        sys.exit(1)

