# -*- coding: utf-8 -*-
from pandas import read_csv
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import numpy as np
import math
import sys
import datetime

# receive the parameters by command line
filename = str(sys.argv[1])

# Custom date parser for NASA log format
def date_parser(x):
    return datetime.datetime.strptime(x, '%d/%b/%Y:%H:%M:%S')

#read the csv file
dataset = read_csv(filename, header=0, parse_dates=[0], index_col=0, date_parser=date_parser)

# split into train and test sets
X = dataset.values
X = X.astype('float32')
size = int(len(X) * 0.67)
train, test = X[0:size], X[size:len(X)]

# Get the index for dates
dataset_dates = dataset.index
train_dates = dataset_dates[0:size]
test_dates = dataset_dates[size:len(X)]

predictions = list()

timer = start = datetime.datetime.now() 
# walk-forward validation
for t in range(len(test)):
    start = datetime.datetime.now()
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    train_end_idx = size + t
    current_train_data = dataset.iloc[0:train_end_idx]
    
    df_prophet = pd.DataFrame({
        'ds': current_train_data.index,
        'y': current_train_data.values.flatten()
    })
    
    # Create and fit model
    model = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
    model.fit(df_prophet)
    
    # Make forecast for next time step
    future = pd.DataFrame({'ds': [test_dates[t]]})
    forecast = model.predict(future)
    
    yhat = forecast['yhat'].values[0]
    predictions.append(yhat)
    
    print(datetime.datetime.now() - start)

# evaluate forecasts
yhat = np.asarray(predictions, dtype=np.float32)
y_test = test.flatten()

print("Timer: ", datetime.datetime.now() - timer)
score = mean_squared_error(y_test, yhat)
r2 = r2_score(y_test, yhat)
print('R2: %.2f, Testscore: %.2f MSE (%.2f RMSE)' %(r2, score, math.sqrt(score)))

# Save into a file the configuration and evaluation executed
f=open("Prophet_Execuções.txt", "a")
f.write("\n python prophet.py %s - R2: %.2f, score: %.2f MSE (%.2f RMSE)" %(filename, r2, score, math.sqrt(score))) 
f.close()
