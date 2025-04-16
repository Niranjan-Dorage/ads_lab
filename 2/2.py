import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("data.csv")  # Replace with your actual file path

# 🔧 Step 1: Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Check column names to confirm
print("🧾 Columns:", df.columns.tolist())

# Replace bad entries with NaN
df.replace(["N/A", "Not Available", "n/a", "NA", ""], np.nan, inplace=True)

# 🔍 Check for correct column name
# It should be 'what_is_your_cgpa?' now
cgpa_col = 'what_is_your_cgpa?'

# Convert CGPA column to numeric
df[cgpa_col] = pd.to_numeric(df[cgpa_col], errors='coerce')

# 📉 Step 2: Remove outliers using IQR
Q1 = df[cgpa_col].quantile(0.25)
Q3 = df[cgpa_col].quantile(0.75)
IQR = Q3 - Q1
filtered_df = df[~((df[cgpa_col] < (Q1 - 1.5 * IQR)) | (df[cgpa_col] > (Q3 + 1.5 * IQR)))]

# 🧩 Step 3: Impute missing CGPA with mean
filtered_df[cgpa_col].fillna(filtered_df[cgpa_col].mean(), inplace=True)

# ✅ Cleaned data preview
print("\n✅ Cleaned Data Preview:")
print(filtered_df[[cgpa_col, 'age', 'choose_your_gender']].head())
