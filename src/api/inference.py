import joblib
import pandas as pd
from datetime import datetime
from schemas import HousePredictionRequest, PredictionResponse
from typing import List

MODEL_PATH="models/trained/house_price_model.pkl"
PREPROCESSOR_PATH="models/trained/preprocessor.pkl"

try:
    model=joblib.load(MODEL_PATH)
    preprocessor=joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading Model or Preprocessor: {str(e)}")

def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """
    Predict house price based on input feature
    """
    input_data=pd.DataFrame(data=[request.dict()])
    input_data['house_age']=datetime.now().year-input_data['year_built']
    input_data['bed_bath_ratio']=input_data['bedrooms']/input_data['bathrooms']
    input_data['price_per_sqft']=0

    processed_features=preprocessor.transform(X=input_data)

    predicted_price=model.predict(X=processed_features)[0]
    predicted_price=round(float(predicted_price),2)
    confidence_interval=[round(float(predicted_price*0.9),2), round(float(predicted_price*1.1),2)]

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        features_importance={},
        prediction_time=datetime.now().isoformat()
    )

def batch_predict(requests:List[HousePredictionRequest]) -> list[float]:
    """
    Performs Batch Predictions
    """
    input_data=pd.DataFrame([req.dict() for req in requests])
    input_data['house_age']=datetime.now().year-input_data['year_built']
    input_data['bed_bath_ratio']=input_data['bedrooms']/input_data['bathrooms']
    input_data['price_per_sqft']=0

    processed_features=preprocessor.transform(X=input_data)

    predicted_prices=model.predict(X=processed_features).tolist()
    responses: List[PredictionResponse] = []
    
    for price in predicted_prices:
        predicted_price = round(float(price), 2)
        confidence_interval = [
            round(float(predicted_price * 0.9), 2), 
            round(float(predicted_price * 1.1), 2)
        ]
        
        responses.append(
            PredictionResponse(
                predicted_price=predicted_price,
                confidence_interval=confidence_interval,
                features_importance={}, # Corrected name from previous step
                prediction_time=datetime.now().isoformat()
            )
        )
        
    return responses # Return the list of structured responses
                             
                         
    