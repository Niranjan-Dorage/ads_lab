import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, t, sem

# Load data
df = pd.read_csv("data.csv")  # Change filename

# --- CLEAN CGPA ---

def extract_midpoint(cgpa_range):
    if isinstance(cgpa_range, str) and '-' in cgpa_range:
        parts = cgpa_range.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return np.nan
    try:
        return float(cgpa_range)
    except:
        return np.nan

df["CGPA"] = df["What is your CGPA?"].apply(extract_midpoint)

# --- CENTRAL TENDENCY ---
print("📊 Central Tendency of CGPA:")
print("Mean   :", df["CGPA"].mean())
print("Median :", df["CGPA"].median())
print("Mode   :", df["CGPA"].mode()[0])

# --- DISPERSION ---
print("\n📈 Nature of Dispersion:")
print("Standard Deviation:", df["CGPA"].std())
print("Variance          :", df["CGPA"].var())
print("Min               :", df["CGPA"].min())
print("Max               :", df["CGPA"].max())
print("Range             :", df["CGPA"].max() - df["CGPA"].min())
print("IQR               :", df["CGPA"].quantile(0.75) - df["CGPA"].quantile(0.25))

# --- VISUALIZATION ---

# Box Plot
plt.figure(figsize=(6, 4))
sns.boxplot(y=df["CGPA"])
plt.title("Box Plot of CGPA")
plt.ylabel("CGPA")
plt.grid(True)
plt.show()

# Scatter Plot (Age vs CGPA)
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df["Age"], y=df["CGPA"], hue=df["Choose your gender"])
plt.title("Scatter Plot: Age vs CGPA")
plt.xlabel("Age")
plt.ylabel("CGPA")
plt.grid(True)
plt.show()

# --- INFERENTIAL STATISTICS ---

# T-test between Male & Female CGPA
group_m = df[df["Choose your gender"] == "Male"]["CGPA"]
group_f = df[df["Choose your gender"] == "Female"]["CGPA"]
t_stat, p_val = ttest_ind(group_m.dropna(), group_f.dropna())

print("\n📌 T-Test: Male vs Female CGPA")
print(f"T-statistic = {t_stat:.3f}, P-value = {p_val:.4f}")
if p_val < 0.05:
    print("✅ Significant difference between male and female CGPA")
else:
    print("❌ No significant difference between male and female CGPA")

# Confidence Interval (95%) for CGPA
mean = df["CGPA"].mean()
n = df["CGPA"].count()
s_error = sem(df["CGPA"].dropna())
ci = t.interval(0.95, n - 1, loc=mean, scale=s_error)

print(f"\n📈 95% Confidence Interval for CGPA: {ci}")
