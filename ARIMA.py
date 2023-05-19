import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import json
import math

#imports and other necessary files / data

dates = np.load('dates.npy', allow_pickle=True).tolist()
dates.reverse()
tags = np.load('labels.npy', allow_pickle=True)[:-1]

with open('SalesPlotData.json') as fp:
    sales_data_all = json.load(fp)

# Arima prediction function, takes item name as input
# returns actual test sales value and series of predictions up until and including the prediction for the date the test value is from

def ARIMA_predict(item):
    item_data = sales_data_all[item]
    item_data.reverse()

    x = dates[:-2]
    y = item_data[:-2]

    
    actual = item_data[-2]

    sales_data = pd.Series(y)

    # Defining ranges for parameters (needed for tuning)
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)

    # Initializing variables
    best_params = None
    best_metric = float('inf')

    # Perform grid search
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            
            arima_model = sm.tsa.ARIMA(sales_data, order=(p, d, q))
            arima_model_fit = arima_model.fit()
            metric = arima_model_fit.aic
            
            if metric < best_metric:
                best_params = (p, d, q)
                best_metric = metric
        except:
            continue

    # Create an ARIMA model with the desired order (p, d, q) and best parameters
    arima_model = sm.tsa.ARIMA(sales_data, order=best_params)

    arima_model_fit = arima_model.fit()

    future_predictions = arima_model_fit.predict(end = 113)
    return (actual, future_predictions)

#Function to cycle through all items, predicting them and storing MSE and total error

def predict_all():
    count = 0
    total_squared_error = 0

    for item in tags:
        
        count += 1
        print(f'{count}/{len(tags)} ---------------------- {(count/len(tags))*100}%')
        
        actual, predicted = ARIMA_predict(item)
        predicted = predicted.iloc[-3]
        total_squared_error += ((float(actual)-float(predicted))**2)
        print(f'actual: {actual}, predicted: {predicted}')

    MSE = total_squared_error / count

    print(total_squared_error) 
    print(MSE)                  

    with open('ARIMA_RESULTS.txt', 'w') as fp:
        fp.write(f'total squared error: {total_squared_error} \nmean squared error: {MSE}')

#predict_all()
