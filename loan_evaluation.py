import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\purna\OneDrive\Desktop\Project2_LoanDefault')
print("📁 Working folder set!")

print("Loading predictions...")
df = pd.read_csv('loan_predictions.csv')
print(f"Loaded {len(df):,} rows")
print(f"Columns: {list(df.columns[-4:])}")

print("\n Confusion Matrix Analysis:")

y_actual    = df['actual_default']
y_predicted = df['predicted_default']
y_probs     = df['default_probability']

cm = confusion_matrix(y_actual, y_predicted)
tn, fp, fn, tp = cm.ravel()

print(f"""
┌─────────────────────────────────────────┐
│           CONFUSION MATRIX              │
├─────────────────┬───────────────────────┤
│                 │     PREDICTED         │
│                 │  Safe   │  Default    │
├─────────────────┼─────────┼─────────────┤
│ ACTUAL  Safe    │ {tn:>7,} │ {fp:>9,}   │
│ ACTUAL  Default │ {fn:>7,} │ {tp:>9,}   │
└─────────────────┴─────────┴─────────────┘
""")

accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * (precision * recall) / (precision + recall)
auc       = roc_auc_score(y_actual, y_probs)

print(f"Model Performance Metrics:")
print(f"ROC-AUC   : {auc:.4f}  → overall discrimination ability")
print(f"Accuracy  : {accuracy:.4f}  → overall correct predictions")
print(f"Precision : {precision:.4f}  → of flagged loans how many truly defaulted")
print(f"Recall    : {recall:.4f}  → of actual defaults how many we caught")
print(f"F1 Score  : {f1:.4f}  → balance between precision and recall")

# ── BUSINESS IMPACT ───────────────────────────────────────
print("\n💰 Business Impact Analysis:")
avg_loan = df['loan_amnt'].mean()

print(f"Average loan amount      : ${avg_loan:,.2f}")
print(f"True Positives (caught)  : {tp:,} defaults caught")
print(f"False Negatives (missed) : {fn:,} defaults missed")
print(f"False Positives (alarms) : {fp:,} safe loans flagged")

money_saved  = tp * avg_loan
money_missed = fn * avg_loan
money_lost   = fp * avg_loan * 0.05  

print(f"\n 💚 Money saved by catching defaults : ${money_saved:,.0f}")
print(f"🔴 Money at risk from missed defaults: ${money_missed:,.0f}")
print(f"🟡 Opportunity cost of false alarms  : ${money_lost:,.0f}")
print(f"\n NET BENEFIT: ${money_saved - money_lost:,.0f}")

# ── ROC CURVE ─────────────────────────────────────────────
print("\n📈 Generating ROC Curve chart...")

fpr, tpr, thresholds = roc_curve(y_actual, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#1A56DB', linewidth=2,
         label=f'Random Forest (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--',
         linewidth=1, label='Random Guessing (AUC = 0.50)')
plt.fill_between(fpr, tpr, alpha=0.1, color='#1A56DB')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve — Loan Default Prediction Model', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.close()
print("ROC curve saved as roc_curve.png")

# ── FEATURE IMPORTANCE CHART ──────────────────────────────
print("📊 Generating Feature Importance chart...")

features = [
    'grade', 'int_rate', 'loan_to_income', 'dti',
    'payment_to_income', 'home_ownership', 'installment',
    'loan_amnt', 'emp_length', 'monthly_income'
]
importances = [
    0.346941, 0.326746, 0.081651, 0.050487,
    0.046823, 0.029965, 0.023189, 0.022524,
    0.014485, 0.011419
]

colors = ['#1A56DB' if i < 3 else '#93C5FD' for i in range(len(features))]

plt.figure(figsize=(10, 6))
bars = plt.barh(features[::-1], importances[::-1], color=colors[::-1])
plt.xlabel('Feature Importance Score', fontsize=12)
plt.title('Top 10 Features Driving Loan Default\n(Random Forest)', fontsize=14)
plt.axvline(x=0.05, color='red', linestyle='--',
            alpha=0.5, label='5% threshold')
plt.legend()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("Feature importance saved as feature_importance.png")

# ── CREATE POWER BI SUMMARY FILES ────────────────────────
print("\n📤 Creating Power BI ready files...")

# File 1: Risk tier summary
risk_summary = df.groupby('risk_tier').agg(
    total_loans        = ('actual_default', 'count'),
    actual_defaults    = ('actual_default', 'sum'),
    avg_loan_amnt      = ('loan_amnt', 'mean'),
    avg_int_rate       = ('int_rate', 'mean'),
    avg_dti            = ('dti', 'mean'),
    avg_default_prob   = ('default_probability', 'mean')
).round(2).reset_index()

risk_summary['default_rate_pct'] = (
    risk_summary['actual_defaults'] /
    risk_summary['total_loans'] * 100
).round(2)

risk_summary.to_csv('powerbi_risk_summary.csv', index=False)
print("✅ Saved powerbi_risk_summary.csv")
print(risk_summary.to_string(index=False))

# File 2: Grade level summary
grade_summary = df.groupby('grade').agg(
    total_loans      = ('actual_default', 'count'),
    actual_defaults  = ('actual_default', 'sum'),
    avg_loan_amnt    = ('loan_amnt', 'mean'),
    avg_int_rate     = ('int_rate', 'mean'),
    avg_default_prob = ('default_probability', 'mean')
).round(2).reset_index()

grade_summary['default_rate_pct'] = (
    grade_summary['actual_defaults'] /
    grade_summary['total_loans'] * 100
).round(2)

grade_summary.to_csv('powerbi_grade_summary.csv', index=False)
print("\n✅ Saved powerbi_grade_summary.csv")

# File 3: Interest rate buckets
df['rate_bucket'] = pd.cut(
    df['int_rate'],
    bins   = [0, 8, 12, 16, 20, 25, 100],
    labels = ['Below 8', '8 to 12', '12 to 16',
              '16 to 20', '20 to 25', 'Above 25']
)

rate_summary = df.groupby('rate_bucket').agg(
    total_loans     = ('actual_default', 'count'),
    actual_defaults = ('actual_default', 'sum'),
    avg_loan_amnt   = ('loan_amnt', 'mean'),
    avg_dti         = ('dti', 'mean')
).round(2).reset_index()

rate_summary['default_rate_pct'] = (
    rate_summary['actual_defaults'] /
    rate_summary['total_loans'] * 100
).round(2)

rate_summary.to_csv('powerbi_rate_summary.csv', index=False)
print("Saved powerbi_rate_summary.csv")

print("\n All files ready for Power BI!")
print("\nFiles created in your project folder:")
print("loan_predictions.csv      → main predictions")
print("powerbi_risk_summary.csv  → risk tier analysis")
print("powerbi_grade_summary.csv → grade level analysis")
print("powerbi_rate_summary.csv  → interest rate analysis")
print("roc_curve.png             → model performance chart")
print("feature_importance.png    → feature importance chart")




