#import modules
import os 
import pandas as pd
import requests
from dotenv import load_dotenv
import sqlite3

load_dotenv()  # take environment variables from .env.

class StockDataApi:
    
    def __init__(self, api_key=os.getenv('API_KEY')):
        
        self.__api_key = api_key
        
        
        
    def Get_Data(self, ticker, outputsize):
        
        #get url
        url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=1day&outputsize={outputsize}&apikey={self.__api_key}"
        #send request to api
        response = requests.get(url)
        #get json data
        response_json = response.json()
        if "values" not in response_json.keys():
            raise Exception(
                f"Invalid API call check that ticker symbol {ticker} is correct"
            )
        #get stock data
        stock_data = response_json["values"]
        #convert to dataframe
        df_stock = pd.DataFrame.from_dict(stock_data, dtype=float)
        #rename 'datetime' column to 'date'
        df_stock.rename(columns={"datetime":"date"}, inplace= True)
        #convert "date" to datetime
        df_stock["date"] = pd.to_datetime(df_stock["date"])
        #convert "date" to index
        df_stock.set_index("date", inplace=True)
        return df_stock
    
#create sql repository    
class SQLRepository:
    
    
    def __init__(self, connection, ticker):
        
        self.connection = connection
        self.ticker = ticker
    
    
     
     #insert data into sqlite table   
    def insert_table(self, table_name, records, if_exists='fail'):
          
        n_inserted = records.to_sql(name=table_name, con=self.connection, if_exists=if_exists)
            
        return{
            "transaction_successful":True,
            "records_inserted":n_inserted}
    
    
    #extract data from sqlite table
    def read_table(self, table_name, limit=100):
        query = f"""
        SELECT * FROM {table_name} LIMIT {limit};
        """
        df = pd.read_sql(query, con=self.connection, index_col="date")
        df.index = pd.to_datetime(df.index)
        return df
        
    
        