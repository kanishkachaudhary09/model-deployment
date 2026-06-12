from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

app = FastAPI(
    title="Iris Prediction API",
    version="1.0"
)

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.post("/predict")
def predict(data: IrisInput):

    features = np.array([
        [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]
    ])

    prediction = model.predict(features)[0]

    classes = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    return {
        "prediction": classes[int(prediction)]
    } 
