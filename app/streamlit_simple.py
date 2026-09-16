import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the src directory to path so we can import DataPreprocessor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the DataPreprocessor class
try:
    from preprocessing import DataPreprocessor
except ImportError:
    # If import fails, define the class here
    class DataPreprocessor:
        def __init__(self):
            self.label_encoders = {}
            self.scaler = None
            self.feature_columns = None
            
        def encode_categorical(self, df):
            from sklearn.preprocessing import LabelEncoder
            import numpy as np
            categorical_cols = ['protocol_type', 'service', 'flag']
            df_encoded = df.copy()
            
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
            
            return df_encoded
        
        def scale_features(self, df, fit=True):
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col not in ['label', 'binary_label', 'difficulty']]
            
            if fit:
                self.scaler = StandardScaler()
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Call fit=True first.")
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            
            return df
        
        def prepare_data(self, df, target_col='binary_label', fit_scaler=True):
            df_encoded = self.encode_categorical(df)
            df_scaled = self.scale_features(df_encoded, fit=fit_scaler)
            X = df_scaled.drop(columns=[target_col, 'label', 'difficulty'])
            y = df_scaled[target_col]
            
            if fit_scaler:
                self.feature_columns = X.columns.tolist()
            
            return X, y
        
        def prepare_prediction_data(self, df):
            df_encoded = self.encode_categorical(df)
            df_scaled = self.scale_features(df_encoded, fit=False)
            
            if self.feature_columns:
                df_scaled = df_scaled[self.feature_columns]
            
            return df_scaled

st.set_page_config(page_title="NIDS", page_icon="🛡️", layout="wide")

st.title("🛡️ Network Intrusion Detection System")

# Check what models are available
st.sidebar.title("📁 Model Status")

# List all files in models directory
try:
    model_files = os.listdir('models')
    st.sidebar.write("Available model files:")
    for f in model_files:
        if f.endswith('.pkl'):
            st.sidebar.write(f"✅ {f}")
except:
    st.sidebar.error("No models directory found")

# Try to load a model
model = None
preprocessor = None

# Try different possible model names
possible_models = [
    'best_model.pkl',
    'xgb_smote.pkl', 
    'xgb_balanced.pkl',
    'random_forest_model.pkl',
    'rf_balanced.pkl'
]

for model_name in possible_models:
    try:
        if os.path.exists(f'models/{model_name}'):
            model = joblib.load(f'models/{model_name}')
            preprocessor = joblib.load('models/preprocessor.pkl')
            st.sidebar.success(f"✅ Loaded: {model_name}")
            break
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {model_name}: {e}")

if model is None:
    st.error("❌ No model found. Please train the model first.")
    st.info("Run: python src/models_balanced_fixed.py")
    st.stop()

st.success("✅ Model loaded successfully!")

# Simple prediction form
st.header("🔍 Test a Connection")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Duration", 0, 50000, 0)
    protocol = st.selectbox("Protocol", ['tcp', 'udp', 'icmp'])
    service = st.selectbox("Service", ['http', 'ftp', 'smtp', 'pop3', 'ssh', 'dns', 'telnet'])
    src_bytes = st.number_input("Source Bytes", 0, 1000000, 0)
    dst_bytes = st.number_input("Destination Bytes", 0, 1000000, 0)

with col2:
    flag = st.selectbox("Flag", ['SF', 'S0', 'REJ', 'RSTO', 'RSTR'])
    count = st.number_input("Connection Count", 0, 500, 0)
    srv_count = st.number_input("Service Count", 0, 500, 0)
    same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 0.0)
    land = st.selectbox("Land Attack", [0, 1])

if st.button("🔍 Analyze", type="primary"):
    # Create feature vector with default values for all 41 features
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
    
    # Create input row with all default values
    input_values = [
        duration, protocol, service, flag, src_bytes,
        dst_bytes, land, 0, 0, 0,  # wrong_fragment, urgent, hot
        0, 0, 0, 0, 0,  # num_failed_logins, logged_in, num_compromised, root_shell, su_attempted
        0, 0, 0, 0, 0, 0,  # num_root, num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_host_login
        0,  # is_guest_login
        count, srv_count,  # count, srv_count
        0.0, 0.0, 0.0, 0.0,  # serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate
        same_srv_rate,  # same_srv_rate
        0.0, 0.0, 0.0, 0.0,  # diff_srv_rate, srv_diff_host_rate, dst_host_count, dst_host_srv_count
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # more dst_host features
    ]
    
    # Ensure we have exactly 41 features
    while len(input_values) < 41:
        input_values.append(0)
    input_values = input_values[:41]
    
    # Create DataFrame
    input_df = pd.DataFrame([input_values], columns=feature_columns)
    
    try:
        # Preprocess
        processed = preprocessor.prepare_prediction_data(input_df)
        
        # Predict
        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0]
        
        # Show results
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if pred == 0:
                st.success("✅ NORMAL")
            else:
                st.error("🚨 ATTACK DETECTED")
        
        with col2:
            st.metric("Confidence", f"{max(prob)*100:.1f}%")
        
        with col3:
            st.metric("Risk", "🔴 HIGH" if pred == 1 else "🟢 LOW")
        
        # Show probability chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Normal', 'Attack'], prob, color=['green', 'red'])
        ax.set_ylim(0, 1)
        ax.set_title('Prediction Probabilities')
        for i, v in enumerate(prob):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Please make sure all features are entered correctly.")