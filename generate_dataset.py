"""
generate_dataset.py
--------------------
Generates synthetic citizen profiles and labels each one against ALL 20
TN scheme rules (read from scheme_rules_full.xlsx) to produce a multi-label
dataset: citizens_dataset.csv

Each row = one citizen profile + one 0/1 column per scheme.
"""

import pandas as pd
import random

# ----------------------------------------------------------------------
# STEP 1: Load the structured scheme rules
# ----------------------------------------------------------------------
scheme_rules = pd.read_excel("scheme_rules_full.xlsx")
print(f"Loaded {len(scheme_rules)} scheme rules\n")


# ----------------------------------------------------------------------
# STEP 2: Generate one random citizen profile
# ----------------------------------------------------------------------
def generate_citizen():
    age = random.randint(1, 85)
    gender = random.choice(["Male", "Female"])
    student = "Yes" if age <= 25 and random.random() < 0.5 else "No"

    citizen = {
        "Age": age,
        "Gender": gender,
        "Income": random.randint(10000, 500000),
        "Student": student,
        "GovtSchoolStudent": random.choice(["Yes", "No"]) if student == "Yes" else "No",
        "Farmer": random.choice(["Yes", "No"]),
        "Widow": random.choice(["Yes", "No"]) if gender == "Female" and age >= 18 else "No",
        "Disability": random.choice(["Yes", "No"]),
        "WorkingWoman": random.choice(["Yes", "No"]) if gender == "Female" else "No",
        "SC": random.choice(["Yes", "No"]),
        "ST": random.choice(["Yes", "No"]),
        "LandOwner": random.choice(["Yes", "No"]),
        "Pregnant": random.choice(["Yes", "No"]) if gender == "Female" and 15 <= age <= 45 else "No",
        "TNResident": random.choice(["Yes", "No"]),
        "NoPermanentHouse": random.choice(["Yes", "No"]),
        "BPL": random.choice(["Yes", "No"]),
        "PrimaryEarnerDeceased": random.choice(["Yes", "No"]),
        "GirlChild": "Yes" if gender == "Female" and age < 18 else "No",
        "EnrolledTraining": random.choice(["Yes", "No"]),
    }
    return citizen


# ----------------------------------------------------------------------
# STEP 3: Generic eligibility checker
#   Compares one citizen against one scheme's rule row.
#   A rule value of "Any" or "No Limit" means that field is not checked.
# ----------------------------------------------------------------------
def is_eligible(citizen, rule):
    # Age range
    if not (rule["MinAge"] <= citizen["Age"] <= rule["MaxAge"]):
        return False

    # Income limit
    if rule["IncomeLimit"] != "No Limit":
        if citizen["Income"] > rule["IncomeLimit"]:
            return False

    # Gender
    if rule["Gender"] != "Any" and citizen["Gender"] != rule["Gender"]:
        return False

    # All the Yes/No condition columns
    yes_no_fields = [
        "Student", "GovtSchoolStudent", "Farmer", "Widow", "Disability",
        "WorkingWoman", "SC", "ST", "LandOwner", "Pregnant", "TNResident",
        "NoPermanentHouse", "BPL", "PrimaryEarnerDeceased", "GirlChild",
        "EnrolledTraining",
    ]
    for field in yes_no_fields:
        rule_value = rule[field]
        if rule_value != "Any" and citizen[field] != rule_value:
            return False

    return True


# ----------------------------------------------------------------------
# STEP 4: Generate N citizens and label each against all 20 schemes
# ----------------------------------------------------------------------
NUM_CITIZENS = 8000
rows = []

for _ in range(NUM_CITIZENS):
    citizen = generate_citizen()

    # Check eligibility for every scheme, add as a 0/1 column
    for _, rule in scheme_rules.iterrows():
        col_name = "scheme_" + rule["SchemeName"].replace(" ", "_").replace("(", "").replace(")", "")
        citizen[col_name] = int(is_eligible(citizen, rule))

    rows.append(citizen)

df = pd.DataFrame(rows)
df.to_csv("citizens_dataset.csv", index=False)

print(f"Generated {NUM_CITIZENS} citizen profiles -> citizens_dataset.csv")
print(f"Total columns: {len(df.columns)} "
      f"({len(scheme_rules)} scheme labels + {len(df.columns) - len(scheme_rules)} profile features)\n")

# ----------------------------------------------------------------------
# STEP 5: Quick sanity check - how many citizens qualify for each scheme
# ----------------------------------------------------------------------
scheme_cols = [c for c in df.columns if c.startswith("scheme_")]
print("Eligible citizen counts per scheme:")
print(df[scheme_cols].sum().sort_values(ascending=False))

avg_schemes = df[scheme_cols].sum(axis=1).mean()
print(f"\nAverage number of schemes each citizen qualifies for: {avg_schemes:.2f}")
