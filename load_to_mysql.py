import pandas as pd
import mysql.connector
import os

os.chdir(r'C:\Users\purna\OneDrive\Desktop\Project2_LoanDefault')
print("Working folder set!")

conn = mysql.connector.connect(
    host     = "localhost",
    user     = "root",
    password = "your_password_here",  # <-- REPLACE with your MySQL password
    database = "loan_default"
)
cursor = conn.cursor()
print("Connected to MySQL!")

print("Loading cleaned data...")
df = pd.read_csv('loan_cleaned.csv')
print(f"Loaded {len(df):,} rows")

core_cols = [
    'loan_amnt', 'int_rate', 'installment', 'grade',
    'emp_length', 'home_ownership', 'annual_inc', 'dti',
    'open_acc', 'revol_util', 'total_acc', 'is_default',
    'loan_to_income', 'monthly_income', 'payment_to_income',
    'high_utilization', 'credit_experience'
]
df_mysql = df[core_cols].copy()

df_mysql = df_mysql.where(pd.notnull(df_mysql), None)

print(f"Inserting {len(df_mysql):,} rows into MySQL...")
print("This will take 3-5 minutes, please wait...")

insert_query = """
    INSERT INTO loan_data (
        loan_amnt, int_rate, installment, grade,
        emp_length, home_ownership, annual_inc, dti,
        open_acc, revol_util, total_acc, is_default,
        loan_to_income, monthly_income, payment_to_income,
        high_utilization, credit_experience
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s
    )
"""

batch_size = 1000
total_inserted = 0

for i in range(0, len(df_mysql), batch_size):
    batch = df_mysql.iloc[i:i+batch_size]
    rows = [tuple(row) for _, row in batch.iterrows()]
    cursor.executemany(insert_query, rows)
    conn.commit()
    total_inserted += len(rows)

    # Progress update every 100,000 rows
    if total_inserted % 100000 == 0:
        print(f"Inserted {total_inserted:,} rows so far...")

print(f"\n All {total_inserted:,} rows inserted successfully!")

cursor.execute("SELECT COUNT(*) FROM loan_data")
count = cursor.fetchone()[0]
print(f"MySQL confirms: {count:,} rows in table")

cursor.execute("""
    SELECT
        COUNT(*) AS total,
        SUM(is_default) AS defaults,
        ROUND(AVG(is_default)*100, 2) AS default_rate_pct
    FROM loan_data
""")
result = cursor.fetchone()
print(f"\n Quick Summary:")
print(f"Total rows   : {result[0]:,}")
print(f"Defaults     : {result[1]:,}")
print(f"Default Rate : {result[2]}%")

cursor.close()
conn.close()
print("\n MySQL connection closed.")


