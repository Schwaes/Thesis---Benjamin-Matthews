import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import json

raw_data = pd.read_json('raw.json', orient = 'columns')
tags = np.load('labels.npy', allow_pickle=True)[:-1]
dates = np.load('dates.npy', allow_pickle=True)


# Creates a json of all sales of all items per month
def salesData():
	time_sales_data = {}

	# initialising dictionary with 0 for every month 
	for item in tags:
		print(count)
		count += 1
		x = []
		y = []
		for date in dates:
			if item not in time_sales_data.keys():
				time_sales_data[item] = {date:0}
			else:
				time_sales_data[item][date] = 0

	#iterating over data and adding to dictionary
	for index, row in raw_data.iterrows():
		if row['SB_SKU'] in tags:
			date = f'{row["Date"].year} {row["Date"].month}'
			time_sales_data[row['SB_SKU']][date] += row['Number']

	print(time_sales_data)
	with open('sales_data.json', 'w') as fp:
	    json.dump(time_sales_data, fp)


# Same as sales data, with better formatting to load into app for plotting
def PlotData():
	with open('sales_data.json', 'r') as fp:
	    time_sales_data = json.load(fp)

	sales_data = {}
	count = 0

	for item in tags:
		count += 1
		print(count)
		y = []
		for date in dates:
			y.append(time_sales_data[item][date])
		sales_data[item] = y

	print(type(sales_data))

	with open('SalesPlotData.json', 'w') as fp:
	    json.dump(sales_data, fp)

