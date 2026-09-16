import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def encode_categorical(self, df):
        """Encode categorical features"""
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
        """Scale numerical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude label columns
        numerical_cols = [col for col in numerical_cols if col not in ['label', 'binary_label', 'difficulty']]
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def prepare_data(self, df, target_col='binary_label', fit_scaler=True):
        """Complete data preparation pipeline"""
        print("🔄 Encoding categorical variables...")
        df_encoded = self.encode_categorical(df)
        
        print("🔄 Scaling numerical features...")
        df_scaled = self.scale_features(df_encoded, fit=fit_scaler)
        
        # Separate features and target
        X = df_scaled.drop(columns=[target_col, 'label', 'difficulty'])
        y = df_scaled[target_col]
        
        # Store feature columns
        if fit_scaler:
            self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save the preprocessor"""
        import joblib
        joblib.dump(self, filepath)
        print(f"✅ Preprocessor saved to {filepath}")

if __name__ == "__main__":
    print("="*50)
    print("DATA PREPROCESSING PIPELINE")
    print("="*50)
    
    # Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv('data/processed/KDDTrain_processed.csv')
    test_df = pd.read_csv('data/processed/KDDTest_processed.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare training data
    print("\n🔄 Processing training data...")
    X_train, y_train = preprocessor.prepare_data(train_df, fit_scaler=True)
    print(f"✅ Training features: {X_train.shape}")
    print(f"✅ Training labels: {y_train.shape}")
    
    # Prepare test data
    print("\n🔄 Processing test data...")
    X_test, y_test = preprocessor.prepare_data(test_df, fit_scaler=False)
    print(f"✅ Test features: {X_test.shape}")
    print(f"✅ Test labels: {y_test.shape}")
    
    # Split training data into train and validation
    print("\n🔄 Splitting training data for validation...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"✅ Training split: {X_train_split.shape}")
    print(f"✅ Validation split: {X_val.shape}")
    
    # Save processed data
    print("\n💾 Saving processed data...")
    X_train_split.to_csv('data/processed/X_train.csv', index=False)
    X_val.to_csv('data/processed/X_val.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train_split.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    print("\n" + "="*50)
    print("✅ DATA PREPROCESSING COMPLETE!")
    print("="*50)
    print(f"\n📊 Training samples: {len(X_train_split)}")
    print(f"📊 Validation samples: {len(X_val)}")
    print(f"📊 Test samples: {len(X_test)}")
    print(f"\n📋 Features: {X_train.shape[1]}")
    print(f"📋 Feature columns: {X_train.columns.tolist()}") 
