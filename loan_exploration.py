import pandas as pd
import os
os.chdir(r'C:/Users/purna/OneDrive/Desktop/Project2_LoanDefault/')
print(f"📁 Working folder: {os.getcwd()}")

# ── LOAD THE DATASET ─────────────────────────────────────

cols_we_need = [
    'loan_amnt',        # how much was borrowed
    'int_rate',         # interest rate %
    'annual_inc',       # borrower annual income
    'dti',              # debt-to-income ratio
    'grade',            # loan grade A through G
    'emp_length',       # employment length
    'home_ownership',   # rent / own / mortgage
    'purpose',          # reason for loan
    'loan_status',      # this is our TARGET column
    'installment',      # monthly payment amount
    'open_acc',         # number of open credit accounts
    'revol_util',       # credit utilization %
    'total_acc'         # total credit accounts ever
]

print("Loading dataset...")
df = pd.read_csv(
    r'C:/Users/purna/OneDrive/Desktop/Project2_LoanDefault/accepted_2007_to_2018Q4.csv.gz',
    usecols=cols_we_need,
    low_memory=False,
    compression='gzip'
)

print(f" Dataset loaded! Shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# ── EXPLORE THE DATA ──────────────────────────────────────

# 1. See what loan statuses exist
print("\n Loan Status Distribution:")
print(df['loan_status'].value_counts())

# 2. Check for missing values
print("\n Missing Values per Column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

# 3. Basic statistics
print("\n Basic Statistics:")
print(df[['loan_amnt', 'int_rate', 'annual_inc', 'dti']].describe().round(2))

# 4. Loan grade distribution
print("\n Loan Grade Distribution:")
print(df['grade'].value_counts())

# 5. Preview first 5 rows
print("\n First 5 Rows:")
print(df.head())