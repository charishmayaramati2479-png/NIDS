import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("MULTI-CLASS ATTACK CLASSIFICATION")
print("="*60)

# Load data
train_df = pd.read_csv('data/processed/KDDTrain_processed.csv')
test_df = pd.read_csv('data/processed/KDDTest_processed.csv')

print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")

# Map attack types to categories
attack_category_map = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 
    'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'processtable': 'dos',
    'udpstorm': 'dos', 'mailbomb': 'dos',
    
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 
    'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 
    'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l',
    'warezmaster': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'snmpgetattack': 'r2l',
    'httptunnel': 'r2l', 'named': 'r2l', 'sendmail': 'r2l', 'snmpguess': 'r2l',
    
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r',
    'perl': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
}

# Create category columns
train_df['attack_category'] = train_df['label'].map(attack_category_map)
test_df['attack_category'] = test_df['label'].map(attack_category_map)

# Map categories to numbers
category_mapping = {
    'normal': 0,
    'dos': 1,
    'probe': 2,
    'r2l': 3,
    'u2r': 4
}

train_df['category_label'] = train_df['attack_category'].map(category_mapping)
test_df['category_label'] = test_df['attack_category'].map(category_mapping)

# Remove rows where category is None (unknown)
train_df = train_df.dropna(subset=['attack_category'])
test_df = test_df.dropna(subset=['attack_category'])

print(f"\nAttack category distribution in training:")
print(train_df['attack_category'].value_counts())

# Prepare data
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    train_df[col] = label_encoders[col].fit_transform(train_df[col])
    test_df[col] = label_encoders[col].transform(test_df[col])

# Scale features
scaler = StandardScaler()
numerical_cols = train_df.select_dtypes(include=[np.number]).columns
numerical_cols = [col for col in numerical_cols if col not in ['label', 'binary_label', 'difficulty', 'attack_category', 'category_label']]

train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

# Separate features and target
X_train = train_df.drop(columns=['label', 'binary_label', 'difficulty', 'attack_category', 'category_label'])
y_train = train_df['category_label'].astype(int)

X_test = test_df.drop(columns=['label', 'binary_label', 'difficulty', 'attack_category', 'category_label'])
y_test = test_df['category_label'].astype(int)

print(f"\nTraining features: {X_train.shape}")
print(f"Test features: {X_test.shape}")

# Train XGBoost for multi-class
print("\n🔄 Training XGBoost for multi-class classification...")
xgb_model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
            yticklabels=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])
plt.title('Confusion Matrix - Multi-Class Attack Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('models/multi_class_confusion_matrix.png', dpi=150)
print("📊 Confusion matrix saved!")

# Save model and preprocessors
joblib.dump(xgb_model, 'models/multi_class_model.pkl')
joblib.dump(scaler, 'models/multi_class_scaler.pkl')
joblib.dump(label_encoders, 'models/multi_class_encoders.pkl')
joblib.dump(category_mapping, 'models/category_mapping.pkl')

print("\n✅ Multi-class model saved to 'models/multi_class_model.pkl'")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

print("\n" + "="*60)
print("✅ MULTI-CLASS CLASSIFICATION COMPLETE!")
print("="*60)