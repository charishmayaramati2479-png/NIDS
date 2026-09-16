import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    print(f"\n📊 {model_name} Evaluation:")
    print("-" * 50)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred
    }

def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'models/{model_name.lower()}_confusion_matrix.png', dpi=150)
    plt.show()
    print(f"📊 Confusion matrix saved as '{model_name.lower()}_confusion_matrix.png'")

if __name__ == "__main__":
    print("="*50)
    print("RANDOM FOREST MODEL TRAINING")
    print("="*50)
    
    # Load preprocessed data
    print("\n📂 Loading preprocessed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    print(f"✅ Training samples: {X_train.shape[0]}")
    print(f"✅ Validation samples: {X_val.shape[0]}")
    print(f"✅ Test samples: {X_test.shape[0]}")
    print(f"✅ Features: {X_train.shape[1]}")
    
    # Train Random Forest
    print("\n🔄 Training Random Forest Classifier...")
    start_time = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    print("\n📊 Validation Set Performance:")
    val_metrics = evaluate_model(rf_model, X_val, y_val, "Random Forest (Validation)")
    
    # Evaluate on test set
    print("\n📊 Test Set Performance:")
    test_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest (Test)")
    
    # Plot confusion matrix for test set
    print("\n📊 Plotting confusion matrix...")
    plot_confusion_matrix(y_test, test_metrics['y_pred'], "Random Forest")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    print("="*50)
    print(feature_importance.head(10).to_string(index=False))
    
    # Save feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances - Random Forest')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=150)
    print("\n📊 Feature importance plot saved as 'feature_importance.png'")
    
    # Save model
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    print("\n✅ Model saved to 'models/random_forest_model.pkl'")
    
    print("\n" + "="*50)
    print("✅ RANDOM FOREST TRAINING COMPLETE!")
    print("="*50)
    print(f"\n📊 Final Test Performance:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1-Score:  {test_metrics['f1']:.4f}") 
