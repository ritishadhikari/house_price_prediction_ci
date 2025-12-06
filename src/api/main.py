from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import HousePredictionRequest, PredictionResponse
from inference import predict_price, batch_predict
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator
app=FastAPI(
    title="House Price Prediction API",
    description=(
        "An API for predicting house prices based on various features"
    ),
    version="1.0.0",
    contact={
        "name": "School of Devops",
        "url": "https://schoolofdevops.com",
        "email": "learn@schoolofdevops.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    )

# Initialize and Instrument Prometheus metric
Instrumentator().instrument(app=app).expose(app=app)

@app.get("/health", response_model=dict)
async def health_check():
    return {"status":"healthy","model_loaded":True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: HousePredictionRequest):
    return predict_price(request=request)

@app.post("/predict-batch", response_model=List[PredictionResponse])
async def batch_predict_endpoint(requests:List[HousePredictionRequest]):
    return batch_predict(requests=requests)