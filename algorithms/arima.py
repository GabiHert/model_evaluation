# -*- coding: utf-8 -*-
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import numpy as np
import math
import sys,os
import datetime
# receive the parameters by command line
filename = str(sys.argv[1])
p = int(sys.argv[2])
d = int(sys.argv[3])
q = int(sys.argv[4])
#perc = float(sys.argv[5])
print p,d,q

# Custom date parser for NASA log format
def date_parser(x):
    return datetime.datetime.strptime(x, '%d/%b/%Y:%H:%M:%S')

#read the csv file
dataset = read_csv(filename, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=date_parser)

# split into train and test sets
X = dataset.values
X = X.astype('float32')
size = int(len(X) * 0.67)
train, test = X[0:size], X[size:len(X)]
series = [x for x in train]
predictions = list()

training_size = len(train)

#test with less samples	
#size = int(training_size * (perc /100))
#series = series[size:training_size:]
timer = start = datetime.datetime.now() 
# walk-forward validation
#print 'current, prediction'
for t in range(len(test)):
  start = datetime.datetime.now() 

  model = ARIMA(series, order=(p,d,q))
  model_fit = model.fit(disp=0)
  output = model_fit.forecast()

  print datetime.datetime.now() - start

  yhat = output[0]
  predictions.append(yhat)
  current = test[t]
  series.append(current)
  
  #print('%.2f, %.2f' % (current,yhat))

# evaluate forecasts

print "Timer: ", datetime.datetime.now() - timer
score = mean_squared_error(test, predictions)
r2 = r2_score(test, predictions)
print ('R2: %.2f, Testscore: %.2f MSE (%.2f RMSE)' %(r2,score, math.sqrt(score)))

# Save into a file the configuration and evaluation executed
f=open("ARIMA_Execuções.txt", "a")
f.write("\n python arima.py %s %d %d %d - R2: %.2f, score: %.2f MSE (%.2f RMSE)" %(filename, p,d,q, r2,score, math.sqrt(score))) 

#plot the execution
#pyplot.plot(test)
#pyplot.plot(predictions, color='red')
#pyplot.show()

