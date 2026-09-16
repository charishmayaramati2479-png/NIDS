from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)
CORS(app)

# Load models
try:
    model = joblib.load('models/best_model.pkl')
    multi_model = joblib.load('models/multi_class_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

feature_columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'NIDS API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.json
        
        # Extract features
        features = []
        for col in feature_columns:
            features.append(data.get(col, 0))
        
        # Create DataFrame
        df = pd.DataFrame([features], columns=feature_columns)
        
        # Preprocess
        processed = preprocessor.prepare_prediction_data(df)
        
        # Binary prediction
        binary_pred = model.predict(processed)[0]
        binary_prob = model.predict_proba(processed)[0]
        
        # Multi-class prediction
        multi_pred = multi_model.predict(processed)[0]
        multi_prob = multi_model.predict_proba(processed)[0]
        
        # Map categories
        category_mapping = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
        attack_type = category_mapping.get(multi_pred, 'Unknown')
        
        return jsonify({
            'binary_prediction': int(binary_pred),
            'binary_confidence': float(max(binary_prob)),
            'multi_class_prediction': int(multi_pred),
            'attack_type': attack_type,
            'probabilities': {
                'Normal': float(multi_prob[0]),
                'DoS': float(multi_prob[1]),
                'Probe': float(multi_prob[2]),
                'R2L': float(multi_prob[3]),
                'U2R': float(multi_prob[4])
            },
            'is_attack': bool(binary_pred == 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.json
        connections = data.get('connections', [])
        
        results = []
        for conn in connections:
            features = [conn.get(col, 0) for col in feature_columns]
            df = pd.DataFrame([features], columns=feature_columns)
            processed = preprocessor.prepare_prediction_data(df)
            
            binary_pred = model.predict(processed)[0]
            binary_prob = model.predict_proba(processed)[0]
            
            results.append({
                'features': conn,
                'prediction': int(binary_pred),
                'confidence': float(max(binary_prob)),
                'is_attack': bool(binary_pred == 1)
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)