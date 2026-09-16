import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page config
st.set_page_config(
    page_title="Advanced NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try importing DataPreprocessor
try:
    from preprocessing import DataPreprocessor
except ImportError:
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
                    raise ValueError("Scaler not fitted.")
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            return df
        
        def prepare_prediction_data(self, df):
            df_encoded = self.encode_categorical(df)
            df_scaled = self.scale_features(df_encoded, fit=False)
            if self.feature_columns:
                df_scaled = df_scaled[self.feature_columns]
            return df_scaled

# Load models
@st.cache_resource
def load_models():
    """Load binary and multi-class models"""
    models = {}
    
    # Load binary model
    try:
        models['binary'] = joblib.load('models/best_model.pkl')
        models['preprocessor'] = joblib.load('models/preprocessor.pkl')
        st.sidebar.success("✅ Binary model loaded")
    except:
        st.sidebar.error("❌ Binary model not found")
    
    # Load multi-class model
    try:
        models['multi'] = joblib.load('models/multi_class_model.pkl')
        models['scaler'] = joblib.load('models/multi_class_scaler.pkl')
        models['encoders'] = joblib.load('models/multi_class_encoders.pkl')
        models['category_mapping'] = joblib.load('models/category_mapping.pkl')
        st.sidebar.success("✅ Multi-class model loaded")
    except:
        st.sidebar.warning("⚠️ Multi-class model not found")
    
    return models

models = load_models()

# Title
st.title("🛡️ Advanced Network Intrusion Detection System")
st.markdown("### Multi-Class Attack Classification | Real-Time Detection | Threat Intelligence")

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Mode",
    ["🔍 Single Prediction", "📊 Advanced Analysis", "📈 Model Performance", "📊 Threat Trends", "ℹ️ About"]
)

# Feature columns
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

# ============================================
# PAGE 1: SINGLE PREDICTION
# ============================================
if page == "🔍 Single Prediction":
    st.header("🔍 Network Traffic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.number_input("Duration", 0, 50000, 0)
        protocol = st.selectbox("Protocol", ['tcp', 'udp', 'icmp'])
        service = st.selectbox("Service", ['http', 'ftp', 'smtp', 'pop3', 'ssh', 'dns', 'telnet', 'www'])
        src_bytes = st.number_input("Source Bytes", 0, 1000000, 0)
        dst_bytes = st.number_input("Destination Bytes", 0, 1000000, 0)
    
    with col2:
        flag = st.selectbox("Flag", ['SF', 'S0', 'REJ', 'RSTO', 'RSTR'])
        count = st.number_input("Connection Count", 0, 500, 0)
        srv_count = st.number_input("Service Count", 0, 500, 0)
        same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 0.0)
        land = st.selectbox("Land Attack", [0, 1])
    
    if st.button("🔍 Analyze Connection", type="primary", use_container_width=True):
        # Prepare input
        input_values = [
            duration, protocol, service, flag, src_bytes,
            dst_bytes, land, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            count, srv_count, 0.0, 0.0, 0.0, 0.0, same_srv_rate, 0.0, 0.0,
            0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        input_df = pd.DataFrame([input_values[:41]], columns=feature_columns)
        
        try:
            # Binary prediction
            if 'binary' in models:
                processed = models['preprocessor'].prepare_prediction_data(input_df)
                binary_pred = models['binary'].predict(processed)[0]
                binary_prob = models['binary'].predict_proba(processed)[0]
            
            # Multi-class prediction
            if 'multi' in models:
                # Use the multi-class model
                from sklearn.preprocessing import LabelEncoder
                for col in ['protocol_type', 'service', 'flag']:
                    le = models['encoders'][col]
                    input_df[col] = le.transform(input_df[col])
                
                numerical_cols = input_df.select_dtypes(include=[np.number]).columns
                input_df[numerical_cols] = models['scaler'].transform(input_df[numerical_cols])
                
                multi_pred = models['multi'].predict(input_df)[0]
                multi_proba = models['multi'].predict_proba(input_df)[0]
                
                reverse_mapping = {v: k for k, v in models['category_mapping'].items()}
                attack_type = reverse_mapping.get(multi_pred, 'Unknown')
            
            # Display results
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if binary_pred == 0:
                    st.success("✅ NORMAL")
                else:
                    st.error("🚨 ATTACK DETECTED")
            
            with col2:
                if 'multi' in models:
                    st.metric("Attack Type", attack_type.upper())
            
            with col3:
                st.metric("Confidence", f"{max(binary_prob)*100:.1f}%")
            
            with col4:
                st.metric("Risk", "🔴 HIGH" if binary_pred == 1 else "🟢 LOW")
            
            # Multi-class probabilities
            if 'multi' in models:
                st.subheader("Attack Type Probabilities")
                prob_df = pd.DataFrame({
                    'Attack Type': ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
                    'Probability': multi_proba
                })
                
                fig = px.bar(prob_df, x='Attack Type', y='Probability', 
                            color='Attack Type', 
                            title='Multi-Class Attack Probabilities',
                            color_discrete_map={
                                'Normal': 'green',
                                'DoS': 'orange',
                                'Probe': 'blue',
                                'R2L': 'purple',
                                'U2R': 'red'
                            })
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================
# PAGE 2: ADVANCED ANALYSIS
# ============================================
elif page == "📊 Advanced Analysis":
    st.header("📊 Advanced Threat Analysis")
    st.info("This section shows detailed analysis including SHAP explanations (coming soon)")

# ============================================
# PAGE 3: MODEL PERFORMANCE
# ============================================
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Metrics")
    
    if 'multi' in models:
        # Load test data
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        
        # Predict with multi-class
        y_pred = models['multi'].predict(X_test)
        
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        with col2:
            st.metric("🎯 Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2%}")
        with col3:
            st.metric("🎯 Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2%}")
        with col4:
            st.metric("🎯 F1-Score", f"{f1_score(y_test, y_pred, average='weighted'):.2%}")

# ============================================
# PAGE 4: THREAT TRENDS
# ============================================
elif page == "📊 Threat Trends":
    st.header("📊 Attack Trend Analysis")
    st.info("Visualization of attack patterns over time")

# ============================================
# PAGE 5: ABOUT
# ============================================
else:
    st.header("ℹ️ About Advanced NIDS")
    st.markdown("""
    ### 🛡️ Advanced Network Intrusion Detection System
    
    **Features:**
    - Binary Classification (Normal vs Attack)
    - Multi-Class Classification (Normal, DoS, Probe, R2L, U2R)
    - Interactive Dashboard
    - Real-time Analysis
    
    **Technology Stack:**
    - Python
    - XGBoost
    - Scikit-learn
    - Streamlit
    - Plotly
    
    **Attack Types Detected:**
    - **DoS** - Denial of Service
    - **Probe** - Scanning
    - **R2L** - Remote to Local
    - **U2R** - User to Root
    """)

st.markdown("---")
st.markdown(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")