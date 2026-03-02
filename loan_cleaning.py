import pandas as pd
import numpy as np
import os
os.chdir(r'C:/Users/purna/OneDrive/Desktop/Project2_LoanDefault/')
print("Working folder set!")

cols_we_need = [
    'loan_amnt', 'int_rate', 'annual_inc',
    'dti', 'grade', 'emp_length', 'home_ownership',
    'purpose', 'loan_status', 'installment',
    'open_acc', 'revol_util', 'total_acc'
]
print("Loading dataset...")
df = pd.read_csv(
    r'C:/Users/purna/OneDrive/Desktop/Project2_LoanDefault/accepted_2007_to_2018Q4.csv.gz',
    usecols=cols_we_need,
    low_memory=False,
    compression='gzip'
)

print(f"Loaded {len(df):,} rows")

print("\n Filtering loan statuses...")

keep_statuses = ['Fully Paid', 'Charged Off']
df = df[df['loan_status'].isin(keep_statuses)]

print(f"Rows after status filter: {len(df):,}")
print(df['loan_status'].value_counts())

df['is_default'] = (df['loan_status'] == 'Charged Off').astype(int)

print(f"\n Target Column Created:")
print(df['is_default'].value_counts())
print(f"Default Rate: {df['is_default'].mean()*100:.2f}%")

df.drop('loan_status', axis=1, inplace=True)

print("\n Removing outliers...")

df = df[(df['dti'] >= 0) & (df['dti'] <= 100)]

df = df[(df['annual_inc'] >= 5000) & (df['annual_inc'] <= 500000)]

print(f"Rows after outlier removal: {len(df):,}")

print("\n Handling missing values...")

df['emp_length'] = df['emp_length'].fillna('Unknown')

df['dti'] = df['dti'].fillna(df['dti'].median())

df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())

df.dropna(inplace=True)

print(f"Missing values handled!")
print(f"Remaining missing values: {df.isnull().sum().sum()}")

print("\n✏️ Cleaning text columns...")

emp_length_map = {
    '< 1 year'  : 0,
    '1 year'    : 1,
    '2 years'   : 2,
    '3 years'   : 3,
    '4 years'   : 4,
    '5 years'   : 5,
    '6 years'   : 6,
    '7 years'   : 7,
    '8 years'   : 8,
    '9 years'   : 9,
    '10+ years' : 10,
    'Unknown'   : -1
}
df['emp_length'] = df['emp_length'].map(emp_length_map)

grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
df['grade'] = df['grade'].map(grade_map)

home_map = {
    'OWN'      : 3,
    'MORTGAGE' : 2,
    'RENT'     : 1,
    'OTHER'    : 0,
    'NONE'     : 0,
    'ANY'      : 0
}
df['home_ownership'] = df['home_ownership'].map(home_map)
df['home_ownership'] = df['home_ownership'].fillna(0)

purpose_dummies = pd.get_dummies(df['purpose'], prefix='purpose')
df = pd.concat([df, purpose_dummies], axis=1)
df.drop('purpose', axis=1, inplace=True)

print("Text columns cleaned and converted!")

print("\n Engineering new features...")
df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
df['monthly_income'] = df['annual_inc'] / 12
df['payment_to_income'] = df['installment'] / df['monthly_income']
df['high_utilization'] = (df['revol_util'] > 80).astype(int)
df['credit_experience'] = df['total_acc'] - df['open_acc']

print("New features created:")
print("loan_to_income, payment_to_income,")
print("high_utilization, credit_experience")

print("\n Final Dataset Summary:")
print(f"Total rows    : {len(df):,}")
print(f"Total columns : {df.shape[1]}")
print(f"Default rate  : {df['is_default'].mean()*100:.2f}%")
print(f"Missing values: {df.isnull().sum().sum()}")

print("\n Column List:")
print(list(df.columns))

print("\n Sample of cleaned data:")
print(df.head(3))

df.to_csv('loan_cleaned.csv', index=False)
print("\n Cleaned data saved to loan_cleaned.csv!")