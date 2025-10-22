import pandas as pd
import numpy as np

df = pd.read_csv('mar9_2023_MCMC.csv')
df = df.drop(df.index[0:1000])
df = df.drop(columns=['Unnamed: 0'])
ind = df.lnprobs.idxmax()

def percentiles(a):

	print(str(a.name) + ': ' + str(np.percentile(a, 50)) + ' +/- ' + str(np.percentile(a, 84) - np.percentile(a, 50)) + ', ' + str(np.percentile(a, 50) - np.percentile(a, 16))) 
	print('Upper limit (if applicable): ',str(np.percentile(a,99.7)))

	print('bestfit for ' + str(a.name) + ' is ' + str(a[ind]))
