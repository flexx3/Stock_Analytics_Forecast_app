#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import modules
import os
import sqlite3


import pandas as pd
from plotly import graph_objects as go
import plotly.express as px
from data import SQLRepository, StockDataApi
import sqlite3
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc


# In[ ]:


#verrify assets folder
assets_folder = 'assets'
image_path = os.path.join(assets_folder, 'stock-icon.png')
if not os.path.exists(os.path.dirname(image_path)):
    os.makedirs(os.path.dirname(image_path))
if os.path.exists(image_path):
    print('image file exists')
else:
    print('image file does not exists')


# In[ ]:


#initialize app
app = Dash(__name__,
           assets_folder='assets',
           #external_stylesheets=[dbc.themes.BOOTSTRAP],
           meta_tags=[{'name': 'viewport',
                      'content': 'width=device-width, initial-scale=1.0'}],
           title="Flexx Project")


# In[ ]:


server = app.server


# In[ ]:


app.layout = html.Div([
        html.Div([html.H1("Stock App", style={'text-align':'left'}),
               html.Img(src="./assets/stock-icon.png")
        
    ],  className= 'banner'),
    
    html.Div([
        html.Div([html.Label('Select Stocks'),dcc.Dropdown(id= 'data-dropdown', options=[
            {'label':'MSFT', 'value':'MSFT'},
            {'label':'AAPL', 'value':'AAPL'},
            {'label':'META', 'value':'META'},
            {'label':'AMZN', 'value':'AMZN'},
            {'label':'TSLA', 'value':'TSLA'},
            {'label':'GOOGL', 'value':'GOOGL'},
            {'label':'WMT', 'value':'WMT'},
            {'label':'NVD', 'value':'NVD'}
            
        ], value='AAPL', multi=False, clearable=False)], className='four columns'),
        html.Div([html.Label('Select no of rows of data'),
            dcc.RadioItems(id='data-radioitem',
                                 options=[2500,5000],
                                 value=2500, inline=True)], className='four columns'),
     html.Div([html.Label('Use Fresh Data'), 
              dcc.RadioItems(id='freshdata-radioitem',
                             options=[{'label':'yes','value':'yes'},
                                     {'label':'no','value':'no'}],
                             value='yes', inline=True)], className='four columns')   
          
    ], className='row'),
    html.Div([html.Div([html.Label('Tabular Data')], style={'text-align':'center', 'color':'white'}),
              dash_table.DataTable(data=[], page_size=6, id='data-table1')],
            style={'backgroundColor':'black'}),
    
    html.Div([html.Div([html.Label("Select Chart")], style={'text-align':'center'}),
        dcc.Dropdown(id='pricechart-dropdown', options=[
            {'label':html.Div(['Linechart'], style={'color':'Purple'}), 'value':'linechart'},
            {'label':html.Div(['Candlestickchart'],style={'color':'Gold'}), 'value':'candlestick'},
            {'label':html.Div(['OHLC chart'],style={'color':'Orange'}), 'value':'ohlc'},
            {'label':html.Div(['Daily Returns chart'], style={'color':'DarkGreen'}), 'value':'returns'},
            {'label':html.Div(['50Day-Volatility'], style={'color':'Black'}), 'value':'50d_volatility'},
            {'label':html.Div(['100Day-Volatility'], style={'color':'Red'}), 'value':'100d_volatility'}
        ],value='linechart')
    ], style={'text-align':'left', 'width':'300px'}),
    
    html.Div([
        dcc.Graph(id='price-chart', figure={})
    ], style={'text-align':'center'}),
        html.Div([html.Label('Calculate Volatility'),
            dcc.RadioItems(id='volt-calcinput', options=[
                {'label':'Daily Volatility', 'value':'daily_volatility'},
                {'label':'Annual Volatility', 'value':'annual_volatility'}
            ], value='daily_volatility')
        ]),
    
        html.Div(id='volt-calcoutput'),
    
        html.Div([html.Label("Forecast Stocks Volatility")], style={'text-align':'center'}),
        html.Div([
        html.Div([html.Label('Select Stocks For Forecast'),dcc.Dropdown(id= 'predictions-dropdown', options=[
            {'label':'MSFT', 'value':'MSFT'},
            {'label':'AAPL', 'value':'AAPL'},
            {'label':'META', 'value':'META'},
            {'label':'AMZN', 'value':'AMZN'},
            {'label':'TSLA', 'value':'TSLA'},
            {'label':'GOOGL', 'value':'GOOGL'},
            {'label':'WMT', 'value':'WMT'},
            {'label':'NVD', 'value':'NVD'}
        ], value='AAPL', multi=False, clearable=False)], className='four columns'),
        
        html.Div([html.Label('Select no of rows of data'),
            dcc.RadioItems(id='pred-radioitem',
                                 options=[2500,5000],
                                 value=2500, inline=True)], className='four columns'),
        html.Div([
            dcc.Input(id='forecast-outputdays',
                      type='number',
                     inputMode='numeric', step=5, value=5),
            html.Button(id='submit-button', n_clicks=0, children='Submit')
        ], className='four columns')
    ], className='row'),
    
    html.Div([html.Div([html.Label('Forecast Data')], style={'text-align':'center','color':'silver' }),
              dash_table.DataTable(data=[], page_size=6, id='forecast-table1')]),
    html.Div([html.Div([html.Label("Plot Forecast")],
                       style={'text-align':'center'}),
             dcc.RadioItems(id='forecast-radio',
                            options=[{'label':'yes', 'value':'yes'},
                                    {'label':'no', 'value':'no'}], value='yes', inline=True)],style={'text-align':'center'}),
    html.Div([dcc.Graph(id='forecast-chart', figure={})], style={'align':'center'})

])


# In[ ]:


#function to get data
def wrangle(ticker, n_observations, use_new_data):
    #set up connection to database
    connection = sqlite3.connect(os.getenv('DB_NAME'), check_same_thread=False)
    #instantiate repo FOR SQLRepository
    repo = SQLRepository(ticker=ticker, connection=connection)
    if use_new_data == True:
        #instantiate api repo
        api = StockDataApi()
        #get new data
        new_data = api.Get_Data(ticker= ticker, outputsize=5000)
        #insert into db table
        repo.insert_table(table_name=ticker, records=new_data, if_exists="replace")
    #read table from database
    df = repo.read_table(table_name=ticker, limit=n_observations)
    #create returns column
    df['return'] = df['close'].pct_change()*100
    #set in ascending order
    df.sort_values(by='date', inplace=True)
    #ffill 'nan' values
    df.fillna(method='ffill', inplace=True)
    return df


# In[ ]:


@callback(
    Output(component_id='data-table1', component_property='data'),
    Input(component_id='data-dropdown', component_property='value'),
    Input(component_id='data-radioitem', component_property='value'),
    Input(component_id='freshdata-radioitem', component_property='value')
)

def get_data(ticker, n_observations, use_new_data):
    if use_new_data == 'no':
        use_new_data = False
    elif use_new_data == 'yes':
        use_new_data = True
    data_table = (wrangle(ticker=ticker, n_observations=n_observations, use_new_data=use_new_data)
                  .reset_index()
                  .to_dict('records'))
    return (data_table)


# In[ ]:


@callback(
    Output(component_id='price-chart', component_property='figure'),
    Input(component_id='data-dropdown', component_property='value'),
    Input(component_id='data-radioitem', component_property='value'),
    Input(component_id='pricechart-dropdown', component_property='value'),
    Input(component_id='freshdata-radioitem', component_property='value')
)
def plot_price(ticker, n_observations, chart, use_new_data):
    if use_new_data == 'no':
        use_new_data = False
    elif use_new_data == 'yes':
        use_new_data = True
    data = wrangle(ticker=ticker, n_observations=n_observations, use_new_data=use_new_data)
    candlestick = go.Figure()
    candlestick.add_trace(go.Candlestick(x=data.index,
                             open=data["open"],
                             high=data["high"],
                             low=data['low'],
                             close=data["close"]))
    
    candlestick.update_layout(title={'text' : f"Candlestick Chart for {ticker} stocks",
                                    'x':0.5,
                                    'font': {
                                        'family':'Arial',
                                        'color':'black'
                                    }},
          xaxis_title="Date",
          yaxis_title="Price", width=1200, height=800)
    
    
    ohlc = go.Figure()
    ohlc.add_trace(go.Ohlc(x=data.index,
                         open=data["open"],
                         high=data["high"],
                         low=data['low'],
                         close=data["close"]))
    
    ohlc.update_layout(title={'text' : f"OHLC Chart for {ticker} stocks",
                                    'x':0.5,
                                    'font': {
                                        'family':'Arial',
                                        'color':'black'
                                    }},
      xaxis_title="Date",
      yaxis_title="Price", width=1200, height=800)
    
    
    linechart = go.Figure()
    linechart.add_trace(go.Scatter(x=data.index, y=data["open"], name="open"))
    linechart.add_trace(go.Scatter(x=data.index, y=data["high"], name="high"))
    linechart.add_trace(go.Scatter(x=data.index, y=data["low"], name="low"))
    linechart.add_trace(go.Scatter(x=data.index, y=data["close"], name="close"))
    
    linechart.update_layout(title={'text' : f"Time Series Line Chart for {ticker} stocks",
                                    'x':0.5,
                                    'font': {
                                        'family':'Arial',
                                        'color':'black'
                                    }},
                        xaxis_title="Date", yaxis_title="Price",  xaxis_rangeslider_visible=True,
                            width=1200, height=800)
    
    
    returnschart = go.Figure()
    returnschart.add_trace(go.Scatter(x=data.index, y=data["return"], line=dict(color='lightgreen')))
    returnschart.update_layout(title={'text' : f"Daily Returns for {ticker} stocks",
                                    'x':0.5,
                                    'font': {
                                        'family':'Arial',
                                        'color':'green'
                                    }},
                               xaxis=dict(color='green', title_text='Date'),
                               yaxis=dict(color='green', title_text='Returns'),
                               xaxis_rangeslider_visible=True, width=1200,
                               height=800)
    
    
    rolling_volatility = data["return"].rolling(window=50).std()
    volatilitychart = go.Figure()
    volatilitychart.add_trace(go.Scatter(x=rolling_volatility.index, y=rolling_volatility, line=dict(color='black')))
    volatilitychart.update_layout(title={'text' : f"50-Day Volatility Chart for {ticker} stocks",
                                    'x':0.5,
                                    'font': {
                                        'family':'Arial',
                                        'color':'black'
                                    }},
                                  xaxis=dict(color='black', title_text='Date'),
                                  yaxis=dict(color='black', title_text='50-Day Volatility'),
                                  xaxis_rangeslider_visible=True,
                                  width=1200, height=800)
    
    rolling_volatility_100_data = data["return"].rolling(window=100).std()
    rolling_volatility_100= go.Figure()
    rolling_volatility_100.add_trace(go.Scatter(x=rolling_volatility_100_data.index,
                                                y=rolling_volatility_100_data, line=dict(color='red')))
    rolling_volatility_100.update_layout(title={'text' : f"100-Day Volatility Chart for {ticker} stocks",
                                    'x':0.5,
                                    'font': {
                                        'family':'Arial',
                                        'color':'red'
                                    }},
                                    xaxis=dict(color='red', title_text='Date'),
                                         yaxis=dict(color='red', title_text='100-Day Volatility'),
                                  xaxis_rangeslider_visible=True,
                                  width=1200, height=800)

    if chart == 'linechart':
        return linechart
    elif chart == 'candlestick':
        return candlestick
    elif chart == 'ohlc':
        return ohlc
    elif chart == 'returns':
        return returnschart
    elif chart == '50d_volatility':
        return volatilitychart
    elif chart == '100d_volatility':
        return rolling_volatility_100


# In[ ]:


import numpy as np
@callback(
    Output(component_id='volt-calcoutput', component_property='children'),
    Input(component_id='data-dropdown', component_property='value'),
    Input(component_id='data-radioitem', component_property='value'),
    Input(component_id='volt-calcinput', component_property='value'),
    Input(component_id='freshdata-radioitem', component_property='value')
)
def calc_volatility(ticker, n_observations, volatility_input, use_new_data):
    if use_new_data == 'no':
        use_new_data = False
    elif use_new_data == 'yes':
        use_new_data = True
    data = wrangle(ticker=ticker, n_observations=n_observations, use_new_data=use_new_data)
    daily_volatility = data['return'].std()
    annual_volatility = (data['return'].std())*np.sqrt(252)
    if volatility_input == 'daily_volatility':
        return(f"Daily Volatility for {ticker} stock: {daily_volatility}")
    elif volatility_input == 'annual_volatility':
        return(f"Annual volatility for {ticker} stock: {annual_volatility}")


# In[ ]:


from model import GarchModel
predictions_plot = None
output_data = None
@callback(
    Output(component_id='forecast-table1', component_property='data'),
    Output(component_id='forecast-chart', component_property='figure'),
    Input(component_id='predictions-dropdown', component_property='value'),
    Input(component_id='pred-radioitem', component_property='value'),
    Input(component_id='submit-button', component_property='n_clicks'),
    State(component_id='forecast-outputdays', component_property='value'),
    Input(component_id='forecast-radio', component_property='value'),
    
)

def model_prediction(ticker, n_observations,clicks, n_days, plot_bar):
    connection = sqlite3.connect(os.getenv('DB_NAME'), check_same_thread=False)
    #instantiate repo FOR SQLRepository
    repo = SQLRepository(ticker=ticker, connection=connection)
    model_stock = GarchModel(ticker=ticker, repo=repo, use_new_data=False)
    # Wrangle data
    model_stock.wrangle_data(n_observations=n_observations)
    #fit model
    model_stock.fit(p=1, q=1)
    # Generate prediction from `model_shop`
    prediction = model_stock.predict_volatility(horizon=n_days)
    prediction_frame = pd.DataFrame(list(prediction.items()), columns=['Date', 'Predicted Volatility'])
    predictions_data = prediction_frame.to_dict('records')
    if clicks is None:
        output_data = []
    else:
        output_data = predictions_data
        
    barplot = px.bar(
        x=prediction_frame['Date'],
        y=prediction_frame['Predicted Volatility'],
        color=prediction_frame['Date']
    )
    barplot.update_layout(title={'text' : f"Forecast of Volatility for  {ticker} stocks for the next {n_days} days",
                                 'x':0.5,
                                 'font': {'family':'Arial','color':'black'}},
                          xaxis_title="Date", yaxis_title="Predicted Volatility")
    if plot_bar == 'no':
        predictions_plot = []
        
    elif plot_bar == 'yes':
        predictions_plot = barplot
        
    return(output_data, predictions_plot)
        


# In[ ]:


#run app
if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:





# In[ ]:




