import os
from pathlib import Path
from glob import glob

import joblib
import pandas as pd
from arch import arch_model
from dotenv import load_dotenv
from data import StockDataApi, SQLRepository


class GarchModel:
    def __init__(self, ticker, repo, use_new_data):
        
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.model_directory = os.getenv('MODEL_DIRECTORY')
        
    def wrangle_data(self, n_observations):
        #add new data to database if required
        if self.use_new_data:
            #instantiate an api class
            api = StockDataApi()
            #Get data
            new_data = api.Get_Data(ticker=self.ticker, outputsize=5000)
            #insert into database
            self.repo.insert_table(table_name=self.ticker, records=new_data, if_exists="replace")
        # Pull data from SQL database
        df = self.repo.read_table(table_name=self.ticker, limit=n_observations)
        # Clean data, attach to class as `data` attribute
        df['return'] = df['close'].pct_change()*100
        df.sort_values(by='date', inplace=True)
        df.fillna(method='ffill', inplace=True)

        self.data = df['return']
            
    def fit(self, p, q):
        # Train Model, attach to `self.model`
        self.model = arch_model(self.data, p=p, q=q, rescale=False).fit(disp=0)

    def __clean_prediction(self, predictions):
        
        start = predictions.index[0] + pd.DateOffset(days=1)
        #create date range
        prediction_dates = pd.bdate_range(start=start, periods=predictions.shape[1])
        # Create prediction index labels, ISO 8601 format
        prediction_index = [d.isoformat() for d in prediction_dates]
        # Extract predictions from DataFrame, get square root   
        data = predictions.values.flatten() ** 0.5
        # Combine `data` and `prediction_index` into Series   
        prediction_formatted = pd.Series(data, index=prediction_index)
        # Return Series as dictionary   
        return prediction_formatted.to_dict()
    
    def predict_volatility(self, horizon):
        # Generate variance forecast from `self.model`
        predictions = self.model.forecast(horizon=horizon, reindex=False).variance
        # Format prediction with `self.__clean_predction`
        prediction_formatted = self.__clean_prediction(predictions)
        # Return `prediction_formatted`
        return prediction_formatted   
    
    
    def dump(self):
        
        # Create timestamp in ISO format
        timestamp = pd.Timestamp.now(tz=None)
        #create filepath, including `self.model_directory whilst ensuring that filepath exists`
        filepath = os.path.join(self.model_directory, (f"{self.ticker}.pkl"))
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            
        #save the model 
        joblib.dump(self.model, filepath)
        return filepath
    
    
    
    def load(self):
        
        #create pattern for glob search
        pattern = os.path.join(self.model_directory, f"*{self.ticker}.pkl")
        #use glob to get most recent model
        try:
            model_path = model_path = sorted(glob(pattern)) [-1]
        except IndexError:
            raise Exception(f" no model trained for '{self.ticker}.' ")
        #load model and attach to self.model
        self.model = joblib.load(model_path)
        return self.model
    
    
    
    
    
    
    
    
    
    
    
    