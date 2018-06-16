
# coding: utf-8

# In[20]:

from statsmodels.tsa.arima_model import ARIMA
import pandas as pd 
from datetime import datetime, timedelta
import calendar
import numpy as np 
import time
import warnings

        
def getdataFromFile():
    ddf=pd.read_csv(r'monthly-milk-production-pounds-p.csv')
        #print (ddf)
        #-------------------------end----------
        return ddf

#-----------end of ElasticSearch data retrieval---------
def Timeseriesweek():
    df=getdataFromFile().replace(float(0), np.nan).dropna(how='any').reset_index().drop('index', axis=1)
    df['value'] = df['value'].astype('float64') 
    #print(df)
    #--------------apply algorithms----------------

    df.Timestamp = pd.to_datetime(df.Date,format='%Y-%m') 
    df.index = df.Timestamp 
    #------- remove todats data --------------
    
    #-------------end----------------------
    
    #-------------apply Arima---------------------
    
    model = ARIMA(df['value'], order=(2,1,0))
    model_fit = model.fit(disp=0)
    #print (model_fit.forecast()[0])
    # you can forecast many step ahead by incrementing step parameter
    forecast = model_fit.forecast(steps=1)[0]

    print (forecast)
      
    #-------end--------------------------------------
    
Timeseriesweek()    
  


# In[ ]:



