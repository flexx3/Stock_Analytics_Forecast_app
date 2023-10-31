#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import modules

import os
import sqlite3
from glob import glob

import joblib
import pandas as pd
import requests
from arch.univariate.base import ARCHModelResult
from data import SQLRepository


# In[ ]:


#open connection to database
connection = sqlite3.connect(database=os.getenv('DB_NAME'), check_same_thread=False)
#instantiate sqlrepository from data module
repo = SQLRepository(connection=connection, ticker='MSFT')


# In[ ]:


from model import GarchModel

# Instantiate a `GarchModel`
gm_stock = GarchModel(ticker="MSFT", repo=repo, use_new_data=False)

# Does `gm_ambuja` have the correct attributes?
assert gm_stock.ticker == "MSFT"
assert gm_stock.repo == repo
assert not gm_stock.use_new_data
assert gm_stock.model_directory == os.getenv('MODEL_DIRECTORY')


# In[ ]:


# Instantiate `GarchModel`, use new data
model_stock = GarchModel(ticker="TSLA", repo=repo, use_new_data=True)

# Check that model doesn't have `data` attribute yet
assert not hasattr(model_stock, "data")

# Wrangle data
model_stock.wrangle_data(n_observations=1000)

# Does model now have `data` attribute?
assert hasattr(model_stock, "data")

# Is the `data` a Series?
assert isinstance(model_stock.data, pd.Series)

# Is Series correct shape?
assert model_stock.data.shape == (1000,)

model_stock.data.tail()


# In[ ]:


# Instantiate `GarchModel`, use old data
model_stock = GarchModel(ticker="AAPL", repo=repo, use_new_data=False)

# Wrangle data
model_stock.wrangle_data(n_observations=1000)

# Fit GARCH(1,1) model to data
model_stock.fit(p=1, q=1)

# Does `model_shop` have a `model` attribute now?
assert hasattr(model_stock, "model")

# Is model correct data type?
assert isinstance(model_stock.model, ARCHModelResult)

# Does model have correct parameters?
assert model_stock.model.params.index.tolist() == ["mu", "omega", "alpha[1]", "beta[1]"]

# Check model parameters
model_stock.model.summary()


# In[ ]:


# Generate prediction from `model_shop`
prediction = model_stock.predict_volatility(horizon=5)

# Is prediction a dictionary?
assert isinstance(prediction, dict)

# Are keys correct data type?
assert all(isinstance(k, str) for k in prediction.keys())

# Are values correct data type?
assert all(isinstance(v, float) for v in prediction.values())

prediction


# In[ ]:


# Save `model_shop` model, assign filename
filename = model_stock.dump()

# Is `filename` a string?
assert isinstance(filename, str)

# Does filename include ticker symbol?
assert model_stock.ticker in filename

# Does file exist?
assert os.path.exists(filename)

filename


# In[ ]:


model_stock = GarchModel(ticker="AAPL", repo=repo, use_new_data=False)

# Check that new `model_stock_test` doesn't have model attached
assert not hasattr(model_stock, "model")

# Load model
model_stock.load()

# Does `model_stock_test` have model attached?
assert hasattr(model_stock, "model")

model_stock.model.summary()


# In[ ]:


#get request to hello path
url = "http://127.0.0.1:8000/hello"
response = requests.get(url=url)
print('status code: ', response.status_code)
response.json()


# In[ ]:


#instantiate FitIn and FitOut
from main import FitIn, FitOut

fi = FitIn(ticker= 'MSFT', use_new_data=True, n_observations=2500, p=1, q=1)
print(fi)
fo = FitOut(ticker= 'MSFT', use_new_data= True, n_observations=2500, p=1, q=1, success=True, message='Nicely fitted')
print(fo)


# In[ ]:


#build model with build_model
from main import build_model

#instantiate GarchModel
model_stock = build_model(ticker='TSLA', use_new_data=False)
# Is `SQLRepository` attached to `model_shop`?
assert isinstance(model_stock.repo, SQLRepository)

# Is SQLite database attached to `SQLRepository`
assert isinstance(model_stock.repo.connection, sqlite3.Connection)

# Is `ticker` attribute correct?
assert model_stock.ticker == "TSLA"

# Is `use_new_data` attribute correct?
assert not model_stock.use_new_data
model_stock


# In[ ]:


# URL of `/fit` path
url = 'http://127.0.0.1:8000/fit'

# Data to send to path
json = {
    'ticker':'META',
    'n_observations': 2500,
    'use_new_data': True,
    'p':1,
    'q':1
}
# Response of post request
response = requests.post(url=url, json=json)
# Inspect response
print("response code:", response.status_code)
response.json()


# In[ ]:


from main import PredictIn, PredictOut

pi = PredictIn(ticker="SHOPERSTOP.BSE", n_days=5)
print(pi)

po = PredictOut(
    ticker="SHOPERSTOP.BSE", n_days=5, success=True, forecast={}, message="success"
)
print(po)


# In[ ]:


#get predictions
#url of '/predict' path
url = 'http://127.0.0.1:8000/predict'
# Data to send to path
json = {
    'ticker':'META',
    'n_days':6,
    'use_new_data':False
}
#response of post request
response = requests.post(url=url, json=json)
# Inspect response
print("response code:", response.status_code)
response.json()


# In[ ]:




