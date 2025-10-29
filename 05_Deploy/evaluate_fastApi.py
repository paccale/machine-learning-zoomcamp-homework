import pickle
from fastapi import FastAPI
from pydantic import BaseModel


model_file = 'pipeline_v1.bin'
with open(model_file, 'rb') as f:
    dv, model = pickle.load(f)
    
app = FastAPI()



class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(customer: Lead):
    customer = customer.model_dump()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    return {
        "conversion_probability": float(y_pred)
    }
    


