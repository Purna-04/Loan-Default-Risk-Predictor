import pandas as pd
import os

os.chdir(r'C:\Users\purna\OneDrive\Desktop\Project2_LoanDefault')

df = pd.read_csv('loan_predictions.csv')

# Use numbered labels so Power BI sorts them correctly
df['rate_bucket'] = pd.cut(
    df['int_rate'],
    bins   = [0, 8, 12, 16, 20, 25, 100],
    labels = [
        '1 Below 8',
        '2 8 to 12',
        '3 12 to 16',
        '4 16 to 20',
        '5 20 to 25',
        '6 Above 25'
    ]
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
print(" Saved!")
print(rate_summary[['rate_bucket', 'default_rate_pct']])
