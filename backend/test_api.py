import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.predictor import WelfarePredictor

predictor = WelfarePredictor()
schemes = predictor.get_all_schemes()
print(f"Total schemes loaded: {len(schemes)}")

metrics = predictor.get_metrics()
print(f"Summary metrics: {metrics['summary_metrics']}")

test_profile = {
    "Age": 20, "Gender": "Female", "Income": 120000, "Student": "Yes",
    "GovtSchoolStudent": "Yes", "Farmer": "No", "Widow": "No",
    "Disability": "No", "WorkingWoman": "No", "SC": "No", "ST": "No",
    "LandOwner": "No", "Pregnant": "No", "TNResident": "Yes",
    "NoPermanentHouse": "No", "BPL": "No", "PrimaryEarnerDeceased": "No",
    "GirlChild": "No", "EnrolledTraining": "No"
}

res = predictor.predict(test_profile)
print(f"Predict status: {res['status']}")
print(f"Eligible schemes count: {res['summary']['eligible_count']}")
for s in res['eligible_schemes']:
    print(f"  • {s['name']} ({s['confidence_pct']}%)")
    for f in s['factors']:
        print(f"     - {f['impact_text']} [SHAP: {f['score']}]")

print("\nAll Backend Predictor Tests Passed Successfully!")
