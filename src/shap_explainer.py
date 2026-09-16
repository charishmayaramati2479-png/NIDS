import shap
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SHAPExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        
        # Create explainer
        print("🔄 Creating SHAP explainer...")
        self.explainer = shap.TreeExplainer(model)
        print("✅ SHAP explainer created")
    
    def explain_prediction(self, X_input):
        """Get SHAP explanation for a single prediction"""
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_input)
        
        # Get prediction
        prediction = self.model.predict(X_input)[0]
        probability = self.model.predict_proba(X_input)[0]
        
        return {
            'shap_values': shap_values,
            'prediction': prediction,
            'probability': probability
        }
    
    def plot_waterfall(self, X_input, feature_names):
        """Create waterfall plot for a single prediction"""
        shap_values = self.explainer.shap_values(X_input)
        
        # For multi-class, get the shap values for the predicted class
        if len(shap_values.shape) > 2:
            pred_class = self.model.predict(X_input)[0]
            shap_values_class = shap_values[0][:, :, pred_class]
            shap_values_flat = shap_values_class.flatten()
        else:
            shap_values_flat = shap_values[0]
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_flat,
                base_values=self.explainer.expected_value,
                data=X_input.values[0],
                feature_names=feature_names
            ),
            show=False
        )
        plt.title('Feature Contributions to Prediction')
        plt.tight_layout()
        
        return plt

    def plot_importance(self, X_input, feature_names):
        """Plot SHAP feature importance"""
        shap_values = self.explainer.shap_values(X_input)
        
        # For multi-class
        if len(shap_values.shape) > 2:
            shap_sum = np.abs(shap_values).mean(axis=0).mean(axis=0)
            shap_sum_flat = shap_sum.flatten()
        else:
            shap_sum = np.abs(shap_values).mean(axis=0)
            shap_sum_flat = shap_sum
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': shap_sum_flat
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'].head(10), importance_df['importance'].head(10))
        plt.xlabel('SHAP Importance')
        plt.title('Top 10 Feature Contributions')
        plt.tight_layout()
        
        return plt, importance_df

# Test function
if __name__ == "__main__":
    print("="*60)
    print("SHAP EXPLAINER TEST")
    print("="*60)
    
    # Load model
    try:
        model = joblib.load('models/multi_class_model.pkl')
        X_test = pd.read_csv('data/processed/X_test.csv')
        feature_names = X_test.columns.tolist()
        
        # Take sample
        X_sample = X_test.head(5)
        
        # Create explainer
        explainer = SHAPExplainer(model, X_sample)
        
        # Explain sample
        result = explainer.explain_prediction(X_sample.head(1))
        print(f"\nPrediction: {result['prediction']}")
        print(f"Probability: {result['probability']}")
        print("\n✅ SHAP explainer working!")
        
    except Exception as e:
        print(f"Error: {e}")