import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import statistics
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

#Creating json of raw data - Cleaning 

raw = pd.read_excel('raw data.xlsx')
raw = raw[raw['Brand group'] != 'Verzendkosten']
raw = raw[raw.SB_SKU.str.contains("SB")]
raw.to_json('raw.json', orient = 'columns')

#raw_data = pd.read_json('raw.json', orient = 'columns')

#print(raw_data[:5])
#print(raw_data.columns)

#Creating a dataframe of customer accounts and their purchase history, needed for collaborative filtering

def makeCustomerData():
	customer_data = {}

	for index, row in raw_data.iterrows():
		if row['AFAS id unique customer id'] not in customer_data.keys():
			customer_data[row['AFAS id unique customer id']] = {row['SB_SKU']:row['Number']}
		else:
			if row['SB_SKU'] not in customer_data[row['AFAS id unique customer id']].keys():
				customer_data[row['AFAS id unique customer id']][row['SB_SKU']] = row['Number']
			else:
				customer_data[row['AFAS id unique customer id']][row['SB_SKU']] += row['Number']
	
	customer_df = pd.DataFrame(customer_data)
	customer_df.fillna(0, inplace = True)
	customer_df['colSum'] = customer_df.sum(axis = 1)
	customer_df = customer_df[customer_df['colSum'] > 1]
	customer_df = customer_df.transpose()
	customer_df['rowSum'] = customer_df.sum(axis = 1)
	customer_df = customer_df[customer_df['rowSum'] > 3]
	customer_df.to_pickle('customer_data.pkl')

#Some EDA

def EDA():
	count_1 = 0
	count_2 = 0
	count_3 = 0
	count_4 = 0
	for customer in customer_data.keys():
		if len(customer_data[customer].keys()) > 1:
			count_1 += 1
		if len(customer_data[customer].keys()) > 2:
			count_2 += 1
		if len(customer_data[customer].keys()) > 3:
			count_3 += 1
		if len(customer_data[customer].keys()) > 4:
			count_4 += 1
	print(f'more than 1: {count_1} \nmore than 2: {count_2} \nmore than 3: {count_3} \nmore than 4: {count_4}')


#Pearsonr similarity - needs df from makeCustomerData()
def Pearsonsims(customer_df):

	customer_array = customer_df.to_numpy()
	customer_array = customer_array[:-1, :-1]

	item_similarities = np.zeros((customer_array.shape[1], customer_array.shape[1]))
	for i in range(customer_array.shape[1]):
		print(f'i  {i}/{customer_array.shape[1]}')
		for j in range(customer_array.shape[1]):
			if i != j:
				item_similarities[i][j], _ = pearsonr(customer_array[:,i], customer_array[:,j])

	print(item_similarities)

	with open('item similarities.pkl', 'wb') as file:
		pickle.dump(item_similarities, file)

#Cosine similarities, used for this study, needs df created by makeCustomerData()
def Cosinesims(customer_df):
	co_matrix = customer_df.T.dot(customer_df)
	np.fill_diagonal(co_matrix.values, 0)

	cos_score_df = pd.DataFrame(cosine_similarity(co_matrix))
	cos_score_df.index = co_matrix.index
	cos_score_df.columns = np.array(co_matrix.index)

	cos_score_df.to_pickle('cosine similarities.pkl')
	'''