"""
20180402
Use already-trained tweet classifier on twitter accounts other than Trump and Hillary.
Did the machine learn Trump vs. Hillary, or Trump vs. Not-Trump?
"""

import sys
filename = sys.argv[-1]

username = filename[:-4]

import csv



import pandas as pd
import numpy as np
df = pd.read_csv(filename, delimiter=';', quotechar='"')

# print('pre-drop size:',len(df))
# drop = []
# for i, row in df.iterrows():
# 	if str(row['username']).lower()!=username.lower():
# 		drop.append(i)
# df = df.drop(drop)
# print('post-drop size:',len(df))

report = pd.DataFrame()
report['text'] = df['text']

messages = df['text']

def preprocess(message):
	res = []
	for word in message.lower().split():
		if 'pic.twitter.com' in word:
			res.append('<pic>')
		elif word.startswith('http'):
			res.append('http')
		elif word.startswith('@'):
			res.append('<@mention>')
		elif word[0].isdigit():
			res.append('<num>')
		else:
			res.append(word)
	return res

messages = messages.map(preprocess)
# messages.to_csv(filename[:-4]+'_p.csv')

def apply_dict(message):
	import json
	with open('log/vocab_dict.json') as f:
	    vocab_dict = json.load(f)
	res = []
	for word in message:
		if word in vocab_dict:
			res.append(vocab_dict[word])
		else:
			res.append(0)
	return res

X = messages.map(apply_dict)

def prop_zeros(l):
	"""input: list of int
	returns proportion of zeros"""
	from collections import Counter
	res = Counter(l).get(0)
	if res == None:
		res=0
	return round(res/len(l),2)

report['prop_zeros'] = X.map(prop_zeros)

from keras.preprocessing import sequence
X = sequence.pad_sequences(X, maxlen=43) #maxlen is from trained model
# y = np.ones((len(X))) #ones corresponds to Hillary

### load model and weights
from keras.models import load_model
model = load_model('log/2018-04-04-08-35model.h5')
model.load_weights('log/2018-04-04-08-35weights.h5')

scores = model.predict(X)
prob_hillary, prob_trump = np.split(scores, 2, axis=1)
report['prob_hillary'] = pd.Series(prob_hillary.reshape((prob_hillary.size)))
report['prob_trump'] = pd.Series(prob_trump.reshape((prob_trump.size)))


# scores = model.evaluate(X, y, verbose=0)
print(report)
print('Average proportion of words not in vocab_dict for {} is'.format(username),np.mean(report['prop_zeros']))
print('Average score for {} is'.format(username),np.mean(scores,axis=0),'+/-',np.std(scores,axis=0))

