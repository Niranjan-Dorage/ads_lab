import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("data.csv")

# Convert CGPA column to numeric (if not already)
df["What is your CGPA?"] = pd.to_numeric(df["What is your CGPA?"], errors='coerce')

# Set style
sns.set(style="whitegrid")

# 1. Histogram
plt.figure(figsize=(6,4))
sns.histplot(df["What is your CGPA?"], bins=5, kde=False, color="skyblue")
plt.title("Histogram of CGPA")
plt.xlabel("CGPA")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(y=df["What is your CGPA?"], color="lightgreen")
plt.title("Boxplot of CGPA")
plt.ylabel("CGPA")
plt.tight_layout()
plt.show()

# 3. KDE Plot (Distribution Plot)
plt.figure(figsize=(6,4))
sns.kdeplot(df["What is your CGPA?"], shade=True, color="coral")
plt.title("KDE Plot of CGPA")
plt.xlabel("CGPA")
plt.tight_layout()
plt.show()

# 4. Scatter Plot (Multivariate Visualization)
plt.figure(figsize=(6,4))
sns.scatterplot(x=df["Age"], y=df["What is your CGPA?"], hue=df["Choose your gender"])
plt.title("Scatter Plot of Age vs CGPA by Gender")
plt.xlabel("Age")
plt.ylabel("CGPA")
plt.tight_layout()
plt.show()
