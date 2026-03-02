import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\purna\OneDrive\Desktop\Project2_LoanDefault')
print("Working folder set!")

# Load data
print("Loading cleaned data...")
df = pd.read_csv('loan_cleaned.csv')
print(f"Loaded {len(df):,} rows")

# Prepare features
y = df['is_default']
X = df.drop('is_default', axis=1).select_dtypes(include=[np.number])
print(f"Features: {X.shape[1]} columns")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# Logistic Regression baseline
print("\nTraining Logistic Regression...")
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
lr             = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_probs       = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc         = roc_auc_score(y_test, lr_probs)
lr_preds       = lr.predict(X_test_scaled)
print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")
print(classification_report(y_test, lr_preds,
      target_names=['Safe','Default']))

# Random Forest main model
print("\n Training Random Forest... (3-5 mins)")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    min_samples_split=50, random_state=42,
    n_jobs=-1, verbose=1
)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_probs)
rf_preds = rf.predict(X_test)
print(f"Random Forest ROC-AUC: {rf_auc:.4f}")
print(classification_report(y_test, rf_preds,
      target_names=['Safe','Default']))

# Model comparison
print(f"\n Model Comparison:")
print(f"{'Logistic Regression':<25} {lr_auc:.4f}")
print(f"{'Random Forest':<25} {rf_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, rf_preds)
tn, fp, fn, tp = cm.ravel()
print(f"\n Confusion Matrix:")
print(f"Caught defaults  : {tp:,}")
print(f"Missed defaults  : {fn:,}")
print(f"False alarms     : {fp:,}")
print(f"Precision        : {tp/(tp+fp):.4f}")
print(f"Recall           : {tp/(tp+fn):.4f}")

# Feature importance
print("\n Top 10 Features:")
importance_df = pd.DataFrame({
    'feature'   : X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.head(10).to_string(index=False))

# Export predictions
print("\n Exporting predictions...")
results_df = X_test.copy()
results_df['actual_default']      = y_test.values
results_df['predicted_default']   = rf_preds
results_df['default_probability'] = rf_probs
results_df['risk_tier'] = pd.cut(
    results_df['default_probability'],
    bins=[0, 0.1, 0.3, 0.5, 1.0],
    labels=['Low Risk','Medium Risk','High Risk','Very High Risk']
)
results_df.to_csv('loan_predictions.csv', index=False)
print(f"Saved {len(results_df):,} rows to loan_predictions.csv")
print(results_df['risk_tier'].value_counts())

