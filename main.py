from fastapi import FastAPI

import os
import sqlite3
from pydantic import BaseModel
from model import GarchModel
from data import SQLRepository
from dotenv import load_dotenv
import requests

#create `FitIn` class

class FitIn(BaseModel):
    ticker: str
    use_new_data: bool
    n_observations: int
    p: int
    q: int

        
#create 'FitOut' class        
class FitOut(FitIn):
    success: bool
    message: str    

#create 'PredictIn' class
class PredictIn(BaseModel):
    ticker: str
    n_days: int
    use_new_data: bool    
        
#create 'PredictOut' class
class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str

        
        
#build_model function
def build_model(ticker, use_new_data):
    
    #set connection to database
    connection = sqlite3.connect(os.getenv('DB_NAME'), check_same_thread=False)
    #instantiate sqlrepository    
    repo = SQLRepository(connection=connection, ticker=ticker)
    #instantiate model
    model = GarchModel(ticker=ticker, use_new_data=use_new_data, repo=repo)
    return model

#instantiate fastapi app
app = FastAPI()

# `"/hello" path with 200 status code

@app.get("/hello", status_code=200)
def hello():
    return {'message':'hello world'}

@app.post("/fit", status_code=200, response_model=FitOut)
def fit_model(request:FitIn):
    
    # Create `response` dictionary from `request`
    response = request.dict()
    # Create try block to handle exceptions
    try:
        # Build model with `build_model` function
        model = build_model(ticker=request.ticker, use_new_data=request.use_new_data)
        # Wrangle data
        model.wrangle_data(n_observations=request.n_observations)
        # Fit model
        model.fit(p=request.p, q=request.q)
        # Save model
        filename = model.dump()
        # Add `"success"` key to `response`
        response['success'] = True
        #add 'message' keye to 'response'
        response['message'] = f"trained and saved '{filename}'"
    # Create except block
    except Exception as e:
        # Add 'success' key to 'response
        response['success'] = False
        #Add 'message' key to 'response'
        response['message'] = str(e)
    
    return response
        
#create '/predict' path with 200 status code
@app.post('/predict', status_code=200, response_model=PredictOut)
def predict_model(request: PredictIn):
    #create resonse directory from request
    response = request.dict()
    #create try block to handle exceptions
    try:
        #build model with build_model function
        model = build_model(ticker=request.ticker, use_new_data=request.use_new_data)
        #load stored model
        model.load()
        #generate predictions
        forecast = model.predict_volatility(horizon=request.n_days)
        #add success key to response
        response['success'] = True
        #add forecast key to response
        response['forecast'] = forecast
        #add message key to response
        response['message'] = ""
     #create except block
    except Exception as e:
        #add success key to response
        response['success'] = False
        #add forecast key to response
        response['forecast'] = {}
        #add message key to response
        response['message'] = str(e)
        
        
        
    return response
    
    
    
    
    
    
    