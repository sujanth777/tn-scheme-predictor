"""
data_analysis.py
------------------
Basic data analysis on citizens_dataset.csv for the panel review.
Produces 4 graphs as PNG files:
  1. eligible_counts_per_scheme.png  - bar chart
  2. age_distribution.png            - histogram
  3. income_distribution.png         - histogram
  4. schemes_per_citizen.png         - histogram
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("citizens_dataset.csv")
scheme_cols = [c for c in df.columns if c.startswith("scheme_")]

# 1. How many citizens qualify for each scheme
plt.figure(figsize=(10, 6))
counts = df[scheme_cols].sum().sort_values(ascending=True)
counts.index = [c.replace("scheme_", "").replace("_", " ") for c in counts.index]
counts.plot(kind="barh", color="steelblue")
plt.title("Number of Eligible Citizens per Scheme")
plt.xlabel("Number of Citizens")
plt.tight_layout()
plt.savefig("eligible_counts_per_scheme.png", dpi=150)
plt.close()

# 2. Age distribution
plt.figure(figsize=(8, 5))
df["Age"].plot(kind="hist", bins=20, color="coral", edgecolor="black")
plt.title("Age Distribution of Synthetic Citizens")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("age_distribution.png", dpi=150)
plt.close()

# 3. Income distribution
plt.figure(figsize=(8, 5))
df["Income"].plot(kind="hist", bins=20, color="seagreen", edgecolor="black")
plt.title("Income Distribution of Synthetic Citizens")
plt.xlabel("Annual Income (INR)")
plt.tight_layout()
plt.savefig("income_distribution.png", dpi=150)
plt.close()

# 4. How many schemes does each citizen qualify for
plt.figure(figsize=(8, 5))
schemes_per_citizen = df[scheme_cols].sum(axis=1)
schemes_per_citizen.plot(kind="hist", bins=range(0, schemes_per_citizen.max() + 2),
                          color="mediumpurple", edgecolor="black")
plt.title("Number of Schemes Each Citizen Qualifies For")
plt.xlabel("Number of Eligible Schemes")
plt.tight_layout()
plt.savefig("schemes_per_citizen.png", dpi=150)
plt.close()

print("Saved 4 graphs:")
print(" - eligible_counts_per_scheme.png")
print(" - age_distribution.png")
print(" - income_distribution.png")
print(" - schemes_per_citizen.png")
print(f"\nAverage schemes per citizen: {schemes_per_citizen.mean():.2f}")
