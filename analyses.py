#!/usr/bin/python

#This is a possible solution to the Kaggle challenge at
#http://www.kaggle.com/c/AlgorithmicTradingChallenge/.  I had been hearing great
#things about the Pandas/NumPy/statsmodels toolchain in Python, so I decided to
#give them a try.  This scored a 0.78674 according to Kaggle, which puts me
#between 10th and 11th place.  Much room for improvement exists, so perhaps in
#the future I will refine the model.  I just used an OLS here.  I actually only
#trained on the first 50,000 rows of the training set out of over 700k.  

import numpy as np
import pandas as pd
import statsmodels.api as sm

#A = pd.read_csv('training.csv', nrows=50000, index_col=0, parse_dates = range(8, 205, 4))
#testSet = pd.read_csv('testing.csv', index_col=0, parse_dates = range(8, 205, 4))

X1 = A[['security_id', 'p_tcount', 'p_value', 'trade_vwap', 'trade_volume', 
	'bid48', 'ask48',
	'bid49', 'ask49',
	'bid50', 'ask50']]

testSet1 = testSet[['security_id', 'p_tcount', 'p_value', 'trade_vwap', 'trade_volume', 
	'bid48', 'ask48',
	'bid49', 'ask49',
	'bid50', 'ask50']]

Y = A[['bid51', 'ask51',
	'bid52', 'ask52',
	'bid53', 'ask53',
	'bid54', 'ask54',
	'bid55', 'ask55',
	'bid56', 'ask56',
	'bid57', 'ask57',
	'bid58', 'ask58',
	'bid59', 'ask59',
	'bid60', 'ask60',
	'bid61', 'ask61',
	'bid62', 'ask62',
	'bid63', 'ask63',
	'bid64', 'ask64',
	'bid65', 'ask65',
	'bid66', 'ask66',
	'bid67', 'ask67',
	'bid68', 'ask68',
	'bid69', 'ask69',
	'bid70', 'ask70',
	'bid71', 'ask71',
	'bid72', 'ask72',
	'bid73', 'ask73',
	'bid74', 'ask74',
	'bid75', 'ask75',
	'bid76', 'ask76',
	'bid77', 'ask77',
	'bid78', 'ask78',
	'bid79', 'ask79',
	'bid80', 'ask80',
	'bid81', 'ask81',
	'bid82', 'ask82',
	'bid83', 'ask83',
	'bid84', 'ask84',
	'bid85', 'ask85',
	'bid86', 'ask86',
	'bid87', 'ask87',
	'bid88', 'ask88',
	'bid89', 'ask89',
	'bid90', 'ask90',
	'bid91', 'ask91',
	'bid92', 'ask92',
	'bid93', 'ask93',
	'bid94', 'ask94',
	'bid95', 'ask95',
	'bid96', 'ask96',
	'bid97', 'ask97',
	'bid98', 'ask98',
	'bid99', 'ask99',
	'bid100', 'ask100']]

X1 = sm.add_constant(X1)
testSet1 = sm.add_constant(testSet1)

results = sm.OLS(Y, X1).fit()

predictions = results.predict(testSet1)

outIndexes = testSet.index[0:50000].reshape(50000,1)

csvOutput = np.concatenate((outIndexes, predictions), axis=1)

outHeader = 'row_id,bid51,ask51,bid52,ask52,bid53,ask53,bid54,ask54,bid55,ask55,bid56,ask56,bid57,ask57,bid58,ask58,bid59,ask59,bid60,ask60,bid61,ask61,bid62,ask62,bid63,ask63,bid64,ask64,bid65,ask65,bid66,ask66,bid67,ask67,bid68,ask68,bid69,ask69,bid70,ask70,bid71,ask71,bid72,ask72,bid73,ask73,bid74,ask74,bid75,ask75,bid76,ask76,bid77,ask77,bid78,ask78,bid79,ask79,bid80,ask80,bid81,ask81,bid82,ask82,bid83,ask83,bid84,ask84,bid85,ask85,bid86,ask86,bid87,ask87,bid88,ask88,bid89,ask89,bid90,ask90,bid91,ask91,bid92,ask92,bid93,ask93,bid94,ask94,bid95,ask95,bid96,ask96,bid97,ask97,bid98,ask98,bid99,ask99,bid100,ask100'

np.savetxt('prettyOkay.csv.gz', csvOutput, delimiter=',', fmt='%0.3f', header=outHeader, comments='') 
