import pickle

model_file = 'pipeline_v1.bin'
with open(model_file, 'rb') as f:
    dv, model = pickle.load(f)
    
customer = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]

print(f"Probability of conversion: {y_pred}")

