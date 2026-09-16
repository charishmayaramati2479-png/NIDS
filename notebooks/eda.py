import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
train_df = pd.read_csv('../data/raw/KDDTrain.csv')
test_df = pd.read_csv('../data/raw/KDDTest.csv')

print("="*50)
print("NSL-KDD Dataset Analysis")
print("="*50)

print(f"\n📊 Training Data Shape: {train_df.shape}")
print(f"📊 Test Data Shape: {test_df.shape}")

print(f"\n📋 Columns: {train_df.columns.tolist()}")

print(f"\n❓ Missing Values: {train_df.isnull().sum().sum()}")

# Target Distribution
print("\n🎯 Attack Distribution in Training Data:")
attack_counts = train_df['label'].value_counts()
print(attack_counts)

# Binary Classification
train_df['binary_label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['binary_label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

print("\n🎯 Binary Label Distribution (0=Normal, 1=Attack):")
print(train_df['binary_label'].value_counts())

# Save processed data
train_df.to_csv('../data/processed/KDDTrain_processed.csv', index=False)
test_df.to_csv('../data/processed/KDDTest_processed.csv', index=False)

print("\n✅ Processed data saved to 'data/processed/'")

# Create visualization
plt.figure(figsize=(12, 6))
colors = ['green' if x == 'normal' else 'red' for x in attack_counts.index]
attack_counts.plot(kind='bar', color=colors)
plt.title('Distribution of Attack Types in Training Data', fontsize=16)
plt.xlabel('Attack Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../data/attack_distribution.png', dpi=150)
print("📊 Visualization saved as 'data/attack_distribution.png'")
plt.show()

print("\n✅ EDA Complete!")