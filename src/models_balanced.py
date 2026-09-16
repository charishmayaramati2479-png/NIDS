import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    print("BALANCED DATA & MODEL OPTIMIZATION")
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
    print(f"📊 Training class distribution: {np.bincount(y_train)}")
    print(f"📊 Test class distribution: {np.bincount(y_test)}")
    
    # ============================================
    # METHOD 1: XGBoost with Class Weighting
    # ============================================
    print("\n" + "="*60)
    print("METHOD 1: XGBoost with Class Weighting")
    print("="*60)
    
    print("\n🔄 Training XGBoost with class weights...")
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"📊 Class weights: {weight_dict}")
    
    xgb_weighted = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=weight_dict[1]/weight_dict[0],
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_weighted.fit(X_train, y_train)
    
    print("\n📊 Test Set Performance:")
    test_metrics_xgb = evaluate_model(xgb_weighted, X_test, y_test, "XGBoost_Weighted")
    
    # ============================================
    # METHOD 2: SMOTE Oversampling
    # ============================================
    print("\n" + "="*60)
    print("METHOD 2: SMOTE Oversampling")
    print("="*60)
    
    print("\n🔄 Applying SMOTE...")
    from imblearn.over_sampling import SMOTE
    
    start_time = time.time()
    
    # Apply SMOTE with auto strategy (fixed)
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"✅ After SMOTE - Training: {X_train_smote.shape[0]}")
    print(f"📊 Class distribution after SMOTE: {np.bincount(y_train_smote)}")
    
    # Train XGBoost on SMOTE data
    xgb_smote = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_smote.fit(X_train_smote, y_train_smote)
    
    smote_time = time.time() - start_time
    print(f"✅ SMOTE + Training completed in {smote_time:.2f} seconds")
    
    print("\n📊 Test Set Performance:")
    test_metrics_smote = evaluate_model(xgb_smote, X_test, y_test, "XGBoost_SMOTE")
    
    # ============================================
    # METHOD 3: Ensembling (Multiple Models)
    # ============================================
    print("\n" + "="*60)
    print("METHOD 3: Ensemble Model (Random Forest + XGBoost)")
    print("="*60)
    
    print("\n🔄 Training ensemble...")
    start_time = time.time()
    
    # Train Random Forest with balanced weights
    rf_balanced = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_balanced.fit(X_train, y_train)
    
    # Train XGBoost with balanced weights
    xgb_balanced = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=weight_dict[1]/weight_dict[0],
        random_state=42,
        eval_metric='logloss'
    )
    xgb_balanced.fit(X_train, y_train)
    
    # Ensemble predictions (voting)
    y_pred_rf = rf_balanced.predict(X_test)
    y_pred_xgb = xgb_balanced.predict(X_test)
    y_pred_ensemble = np.where((y_pred_rf + y_pred_xgb) >= 1, 1, 0)
    
    ensemble_time = time.time() - start_time
    print(f"✅ Ensemble training completed in {ensemble_time:.2f} seconds")
    
    # Evaluate ensemble
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    ensemble_precision = precision_score(y_test, y_pred_ensemble)
    ensemble_recall = recall_score(y_test, y_pred_ensemble)
    ensemble_f1 = f1_score(y_test, y_pred_ensemble)
    
    print("\n📊 Test Set Performance (Ensemble):")
    print(f"✅ Accuracy:  {ensemble_accuracy:.4f}")
    print(f"✅ Precision: {ensemble_precision:.4f}")
    print(f"✅ Recall:    {ensemble_recall:.4f}")
    print(f"✅ F1-Score:  {ensemble_f1:.4f}")
    
    ensemble_metrics = {
        'accuracy': ensemble_accuracy,
        'precision': ensemble_precision,
        'recall': ensemble_recall,
        'f1': ensemble_f1,
        'y_pred': y_pred_ensemble
    }
    
    # ============================================
    # COMPARE ALL METHODS
    # ============================================
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Model': ['XGBoost (Weighted)', 'XGBoost (SMOTE)', 'Ensemble (RF+XGB)'],
        'Accuracy': [test_metrics_xgb['accuracy'], test_metrics_smote['accuracy'], ensemble_metrics['accuracy']],
        'Precision': [test_metrics_xgb['precision'], test_metrics_smote['precision'], ensemble_metrics['precision']],
        'Recall': [test_metrics_xgb['recall'], test_metrics_smote['recall'], ensemble_metrics['recall']],
        'F1-Score': [test_metrics_xgb['f1'], test_metrics_smote['f1'], ensemble_metrics['f1']]
    })
    
    print(comparison_df.to_string(index=False))
    
    # ============================================
    # SAVE BEST MODEL
    # ============================================
    # Choose best model based on F1
    if ensemble_metrics['f1'] >= test_metrics_smote['f1'] and ensemble_metrics['f1'] >= test_metrics_xgb['f1']:
        best_model_name = "Ensemble (RF+XGB)"
        best_metrics = ensemble_metrics
        joblib.dump(rf_balanced, 'models/rf_balanced.pkl')
        joblib.dump(xgb_balanced, 'models/xgb_balanced.pkl')
        print(f"\n✅ Ensemble models saved to 'models/rf_balanced.pkl' and 'models/xgb_balanced.pkl'")
    elif test_metrics_smote['f1'] >= test_metrics_xgb['f1']:
        best_model = xgb_smote
        best_model_name = "XGBoost (SMOTE)"
        best_metrics = test_metrics_smote
        joblib.dump(best_model, 'models/best_model.pkl')
    else:
        best_model = xgb_weighted
        best_model_name = "XGBoost (Weighted)"
        best_metrics = test_metrics_xgb
        joblib.dump(best_model, 'models/best_model.pkl')
    
    print(f"✅ Best model ({best_model_name}) saved!")
    
    # Plot confusion matrix for best model
    print("\n📊 Plotting confusion matrix for best model...")
    plot_confusion_matrix(y_test, best_metrics['y_pred'], best_model_name)
    
    print("\n" + "="*60)
    print("✅ MODEL OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"   Test Precision: {best_metrics['precision']:.4f}")
    print(f"   Test Recall:    {best_metrics['recall']:.4f}")
    print(f"   Test F1-Score:  {best_metrics['f1']:.4f}")