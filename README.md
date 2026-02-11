# AI Skin Lesion Analyzer 🔬

An AI-powered web application for analyzing skin lesions using deep learning. This multimodal model combines image analysis with patient metadata to predict whether a skin lesion is benign or malignant.

## ⚠️ Important Disclaimer

**This tool is for educational and informational purposes only and is NOT intended for medical diagnosis. Always consult with qualified healthcare professionals for any skin concerns.**


## 🖥️ Application Screenshots
![Prediction Result](photos/1.jpg)
![Prediction Result](photos/2.jpg)

## 🌟 Features

- **Multimodal Deep Learning**: Combines image data with patient metadata (age, sex, localization)
- **Real-time Predictions**: Instant analysis of uploaded skin lesion images
- **Risk Assessment**: Provides confidence scores and risk levels
- **User-friendly Interface**: Clean, intuitive React-based frontend
- **RESTful API**: Flask backend for easy integration

## 🏗️ Architecture

### Backend
- **Framework**: Flask (Python)
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: PIL (Python Imaging Library)
- **Data Processing**: Pandas, NumPy, Scikit-learn

### Frontend
- **Framework**: React.js
- **HTTP Client**: Axios
- **Styling**: Custom CSS

## 📋 Prerequisites

### Backend Requirements
- Python 3.8 or higher
- pip (Python package manager)

### Frontend Requirements
- Node.js 14.x or higher
- npm or yarn

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/BetulBNY/skin_cancer_multimodal.git
cd skin-cancer-detection
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
# Navigate to backend directory
cd backend

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install flask flask-cors tensorflow pillow numpy pandas scikit-learn
```

#### Required Files

Make sure you have these files in your `backend` directory:
- `final_multimodal_model.h5` - The trained model
- `preprocessor.pkl` - The fitted preprocessor
- `app.py` - Flask application

#### Start the Backend Server

```bash
python app.py
```

The backend will start on `http://localhost:5000`

You should see:
```
Flask uygulaması başlatılıyor..
Model uploaded successfully
Preprocessor uploaded successfully
All components are installed, the application is ready!
 * Running on http://0.0.0.0:5000
```

### 3. Frontend Setup

#### Install Node Dependencies

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install

#### Required Dependencies

The frontend requires these packages (should be in `package.json`):
```json
{
  "dependencies": {
    "react": "^18.x",
    "react-dom": "^18.x",
    "axios": "^1.x"
  }
}
```

#### Start the Frontend Development Server

```bash
npm start


The frontend will start on `http://localhost:3000` and automatically open in your browser.

## 📁 Project Structure

```
skin-cancer-detection/
│
├── backend/
│   ├── app.py                          # Flask application
│   ├── final_multimodal_model.h5       # Trained model
│   ├── preprocessor.pkl                # Data preprocessor
│   
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js                      # Main App component
│   │   ├── App.css                     # App styles
│   │   ├── PredictionForm.js           # Prediction form component
│   │   ├── PredictionForm.css          # Form styles
│   │   ├── index.js                    # Entry point
│   │   └── index.css                   # Global styles
│   └── package.json
│
├── preprocessing/
│   └── __init__.py                     # Preprocessing utilities
│   └── data_preprocessing.py
├── data/
│   └── extracted_files/                # Dataset location
│
└── README.md
```

## 🎯 Usage

1. **Start the Backend**: Make sure the Flask server is running on port 5000
2. **Start the Frontend**: Make sure the React app is running on port 3000
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Upload Image**: Select a skin lesion image (JPEG or PNG)
5. **Enter Patient Data**:
   - Age (0-120)
   - Gender (Male/Female/Other)
   - Lesion location on body
6. **Get Prediction**: Click "Predict" to receive analysis

## 🔧 API Endpoints

### Health Check
```http
GET http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "loaded",
  "preprocessor": "loaded"
}
```

### Prediction
```http
POST http://localhost:5000/predict
Content-Type: multipart/form-data

Body:
- image: <image file>
- age: <number>
- sex: <string>
- localization: <string>
```

Response:
```json
{
  "prediction": "Benign (Healthy)" | "Malignant (Cancerous)",
  "confidence": 0.85,
  "risk_level": "Low" | "Medium" | "High",
  "recommendation": "Follow" | "Consult a doctor"
}
```

## 🧪 Model Information

### Input Features

**Image Data:**
- Input size: 224x224 RGB
- Preprocessing: Normalization (0-1 scale)

**Tabular Data:**
- `age`: Patient age
- `sex`: Patient gender
- `localization`: Lesion location
- `age_group`: Derived feature (0-20, 21-40, 41-60, 61+)
- `loc_mean_age`: Average age for lesion location
- `age_dev_from_loc_mean`: Age deviation from location mean

### Output
- Binary classification: Benign vs Malignant
- Confidence score (0-1)
- Risk level assessment
- Medical recommendation

## 🐛 Troubleshooting

### Backend Issues

**Model not loading:**
- Ensure `final_multimodal_model.h5` exists in the backend directory
- Check file permissions

**Preprocessor error:**
- Verify `preprocessor.pkl` is present
- Ensure scikit-learn version compatibility

**Port already in use:**
```bash
# Change port in app.py:
app.run(host='0.0.0.0', port=5001, debug=True)
```

### Frontend Issues

**CORS errors:**
- Ensure Flask-CORS is installed: `pip install flask-cors`
- Backend must be running before frontend

**Cannot connect to backend:**
- Check backend is running on `http://localhost:5000`
- Verify API endpoint in `PredictionForm.js`

**npm install fails:**
```bash
# Clear npm cache
npm cache clean --force
# Try installing again
npm install
```

## 📊 Dataset

This project uses the HAM10000 dataset (Human Against Machine with 10000 training images).

**Source**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

## 📝 License

This project is for educational purposes. Please ensure compliance with dataset licenses and local regulations regarding medical AI applications.


**Remember**: This is a learning project and should never replace professional medical advice!