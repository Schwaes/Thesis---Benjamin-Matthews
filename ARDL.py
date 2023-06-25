import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import ARDL
import json, math, pickle, itertools
from sklearn.model_selection import GridSearchCV

with open('SalesPlotData.json') as fp:
    sales_data_all = json.load(fp)

with open('cosine similarities.pkl', 'rb') as fp:
    similarities = pickle.load(fp)

tags = np.load('labels.npy', allow_pickle=True)[:-1].tolist()

dates = np.load('dates.npy', allow_pickle=True).tolist()
dates.reverse()

csimilarities = similarities.to_numpy()
k_items = 6


#Takes item name and predicts using ARDL
def ARDL_predict(item):
    item_data = sales_data_all[item]
    item_data.reverse()

    #Get k most similar items by accessing the cosine similarity matrix
    item_ind = tags.index(item)
    arr = csimilarities[item_ind, :]
    ind = np.argpartition(arr, -k_items+1)[-k_items+1:]
    top5 = arr[ind].tolist()
    top5ind = [np.where(arr == i)[0][0] for i in top5]

    similar_items = [tags[i] for i in top5ind]

    #Remove target item, as it always has highest similarity metric
    for val, a in zip(top5, similar_items):
        if a == item:
            top5.remove(val)
            similar_items.remove(a)

    #training data
    x = dates
    y = item_data

    actual = item_data[-2]
    sales_data = pd.Series(y)

    similar_items_data = pd.DataFrame()

    for item in similar_items:
        similar_item_data = sales_data_all[item]
        similar_item_data.reverse()
        similar_item_y = similar_item_data
        similar_items_data[item] = similar_item_y


    model = ARDL(sales_data, 3, similar_items_data, 3)

    modelfit = model.fit()
    future_predictions = modelfit.predict()

    return (actual, future_predictions)

#Function to cycle through all items, predicting them and storing MSE and total error

def predict_all():
    count = 0
    total_squared_error = 0

    for item in tags:
        
        count += 1
        print(f'{count}/{len(tags)} ---------------------- {(count/len(tags))*100:.2f}%')
        
        actual, predicted = ARDL_predict(item)
        predicted = predicted.values[-2]
        total_squared_error += ((float(actual)-float(predicted))**2)
        if actual != 0:
            print(f'actual: {actual}, predicted: {predicted}')

    MSE = total_squared_error / count

    print(total_squared_error) 
    print(MSE)                  

    with open('ARDL_RESULTS.txt', 'w') as fp:
        fp.write(f'total squared error: {total_squared_error} \nmean squared error: {MSE}')

predict_all()
