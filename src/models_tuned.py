import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
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
    plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("MODEL IMPROVEMENT & HYPERPARAMETER TUNING")
    print("="*60)
    
    # Load data
    print("\n📂 Loading preprocessed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    print(f"✅ Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # ============================================
    # MODEL 1: TUNED RANDOM FOREST
    # ============================================
    print("\n" + "="*60)
    print("MODEL 1: TUNED RANDOM FOREST")
    print("="*60)
    
    print("\n🔄 Training with hyperparameter tuning...")
    start_time = time.time()
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_tuned = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf_tuned, 
        param_grid, 
        cv=3, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    tuning_time = time.time() - start_time
    print(f"✅ Tuning completed in {tuning_time:.2f} seconds")
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best cross-validation F1: {grid_search.best_score_:.4f}")
    
    best_rf = grid_search.best_estimator_
    
    print("\n📊 Validation Set Performance:")
    val_metrics_rf = evaluate_model(best_rf, X_val, y_val, "Tuned Random Forest")
    
    print("\n📊 Test Set Performance:")
    test_metrics_rf = evaluate_model(best_rf, X_test, y_test, "Tuned Random Forest")
    
    # ============================================
    # MODEL 2: XGBOOST (Simplified)
    # ============================================
    print("\n" + "="*60)
    print("MODEL 2: XGBOOST")
    print("="*60)
    
    print("\n🔄 Training XGBoost Classifier...")
    start_time = time.time()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train without early stopping (simpler)
    xgb_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    print("\n📊 Validation Set Performance:")
    val_metrics_xgb = evaluate_model(xgb_model, X_val, y_val, "XGBoost")
    
    print("\n📊 Test Set Performance:")
    test_metrics_xgb = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    # ============================================
    # COMPARE MODELS
    # ============================================
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Model': ['Tuned Random Forest', 'XGBoost'],
        'Validation Accuracy': [val_metrics_rf['accuracy'], val_metrics_xgb['accuracy']],
        'Test Accuracy': [test_metrics_rf['accuracy'], test_metrics_xgb['accuracy']],
        'Test Precision': [test_metrics_rf['precision'], test_metrics_xgb['precision']],
        'Test Recall': [test_metrics_rf['recall'], test_metrics_xgb['recall']],
        'Test F1': [test_metrics_rf['f1'], test_metrics_xgb['f1']]
    })
    
    print(comparison_df.to_string(index=False))
    
    # ============================================
    # SAVE BEST MODEL
    # ============================================
    # Choose best model based on test F1
    if test_metrics_xgb['f1'] > test_metrics_rf['f1']:
        best_model = xgb_model
        best_model_name = "XGBoost"
        best_metrics = test_metrics_xgb
    else:
        best_model = best_rf
        best_model_name = "Tuned Random Forest"
        best_metrics = test_metrics_rf
    
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"\n✅ Best model ({best_model_name}) saved to 'models/best_model.pkl'")
    
    # Plot confusion matrix for best model
    print("\n📊 Plotting confusion matrix for best model...")
    plot_confusion_matrix(y_test, best_metrics['y_pred'], best_model_name)
    
    # Feature importance for best model
    print("\n📊 Top 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Save feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 10 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig('models/best_feature_importance.png', dpi=150)
    print("\n📊 Feature importance plot saved as 'best_feature_importance.png'")
    
    print("\n" + "="*60)
    print("✅ MODEL IMPROVEMENT COMPLETE!")
    print("="*60)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"   Test Precision: {best_metrics['precision']:.4f}")
    print(f"   Test Recall:    {best_metrics['recall']:.4f}")
    print(f"   Test F1-Score:  {best_metrics['f1']:.4f}")