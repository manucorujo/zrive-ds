import uvicorn
from fastapi import FastAPI, status
from pydantic import BaseModel
from basket_model.feature_store import FeatureStore
from basket_model.basket_model import BasketModel
from basket_model.exceptions import UserNotFoundException, PredictionException


class User(BaseModel):
    user_id: str


# Create an instance of FastAPI
app = FastAPI()

feature_store = FeatureStore()
basket_model = BasketModel()


# Define a route for the root URL ("/")
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.get("/status")
def read_status():
    return {"status": status.HTTP_200_OK}


@app.post("/predict")
async def predict(user: User):
    try:
        features = feature_store.get_features(user.user_id)
        prediction = basket_model.predict(features)
        return {"prediction": prediction}
    except Exception as exception:
        return {f"error {str(exception)}\n"}


# This block allows you to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
