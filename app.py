from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the trained model and scaler
try:
    model = joblib.load('best_gradient_boosting_model_tuned.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Feature names for validation
FEATURE_NAMES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Feature descriptions for frontend
FEATURE_INFO = {
    'CRIM': {'name': 'Crime Rate', 'description': 'Per capita crime rate by town', 'min': 0.0, 'max': 100.0, 'unit': '%'},
    'ZN': {'name': 'Residential Land', 'description': 'Proportion zoned for lots over 25,000 sq.ft', 'min': 0.0, 'max': 100.0, 'unit': '%'},
    'INDUS': {'name': 'Industrial Area', 'description': 'Proportion of non-retail business acres', 'min': 0.0, 'max': 30.0, 'unit': '%'},
    'CHAS': {'name': 'Charles River', 'description': 'Bounds Charles River (1=Yes, 0=No)', 'min': 0, 'max': 1, 'unit': ''},
    'NOX': {'name': 'Air Quality', 'description': 'Nitric oxides concentration', 'min': 0.3, 'max': 1.0, 'unit': 'ppm'},
    'RM': {'name': 'Average Rooms', 'description': 'Average number of rooms per dwelling', 'min': 3.0, 'max': 10.0, 'unit': 'rooms'},
    'AGE': {'name': 'Old Houses', 'description': 'Proportion built prior to 1940', 'min': 0.0, 'max': 100.0, 'unit': '%'},
    'DIS': {'name': 'Employment Distance', 'description': 'Distance to employment centers', 'min': 1.0, 'max': 15.0, 'unit': 'miles'},
    'RAD': {'name': 'Highway Access', 'description': 'Index of accessibility to highways', 'min': 1, 'max': 24, 'unit': 'index'},
    'TAX': {'name': 'Property Tax', 'description': 'Property tax rate per $10,000', 'min': 150, 'max': 800, 'unit': '$'},
    'PTRATIO': {'name': 'School Quality', 'description': 'Pupil-teacher ratio by town', 'min': 12.0, 'max': 25.0, 'unit': 'ratio'},
    'B': {'name': 'Demographics', 'description': 'Proportion of blacks by town', 'min': 0.0, 'max': 400.0, 'unit': 'index'},
    'LSTAT': {'name': 'Lower Status', 'description': 'Percentage lower status population', 'min': 1.0, 'max': 40.0, 'unit': '%'}
}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Boston House Price Prediction API is running!'
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get feature information for frontend"""
    return jsonify({
        'features': FEATURE_INFO,
        'total_features': len(FEATURE_NAMES)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate and extract features
        features = []
        missing_features = []
        
        for feature in FEATURE_NAMES:
            if feature in data:
                features.append(float(data[feature]))
            else:
                missing_features.append(feature)
        
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing': missing_features
            }), 400
        
        # Create prediction input
        input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Convert to actual price (multiply by 1000 for 1970s prices)
        predicted_price = round(prediction, 2)
        
        # Calculate confidence based on feature values
        confidence = calculate_confidence(features)
        
        return jsonify({
            'prediction': predicted_price,
            'price_formatted': f"${predicted_price:.1f}K",
            'confidence': confidence,
            'input_features': dict(zip(FEATURE_NAMES, features)),
            'model': 'Gradient Boosting Regressor',
            'accuracy': '91.79%'
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def calculate_confidence(features):
    """Calculate prediction confidence based on feature ranges"""
    # Simple confidence calculation based on how typical the input values are
    confidence = 85  # Base confidence
    
    # Check if features are within typical ranges
    typical_ranges = {
        0: (0.0, 20.0),    # CRIM
        1: (0.0, 100.0),   # ZN
        2: (0.0, 25.0),    # INDUS
        4: (0.4, 0.9),     # NOX
        5: (4.0, 8.0),     # RM
        12: (5.0, 30.0)    # LSTAT
    }
    
    for idx, (min_val, max_val) in typical_ranges.items():
        if min_val <= features[idx] <= max_val:
            confidence += 2
        else:
            confidence -= 3
    
    return min(95, max(60, confidence))

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        predictions = []
        
        for item in data.get('batch', []):
            features = [float(item[feature]) for feature in FEATURE_NAMES]
            input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
            prediction = model.predict(input_df)[0]
            
            predictions.append({
                'input': item,
                'prediction': round(prediction, 2),
                'price_formatted': f"${prediction:.1f}K"
            })
        
        return jsonify({
            'predictions': predictions,
            'total_predictions': len(predictions)
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Boston House Price Prediction API...")
    print("📊 Model Accuracy: 91.79%")
    print("🔗 API endpoints available:")
    print("   - GET  /api/health")
    print("   - GET  /api/features") 
    print("   - POST /api/predict")
    print("   - POST /api/batch-predict")
    app.run(debug=True, host='0.0.0.0', port=5000)
