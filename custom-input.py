"""
20180403
Use already-trained tweet classifier on custom inputs
"""

messages = [
'Make America Great Again!',
'I am going to build a wall and make Mexico pay for it!',
'This is so sad.',
'This is gonna be huge!',
"tomorrow , we have the chance to stand up for the america we believe in . rt this if you're voting . <url> <pic>",
"yesterday , khizr khan told the story of a <num> - year-old boy who was bullied at school — until his class watched mr . khan's <pic>",
"women can stop trump . here's how : <url> <pic>"
]

import pandas as pd
import numpy as np

df = pd.DataFrame()

messages = pd.Series(messages)

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

df['messages'] = messages

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

def num_zeros(l):
	from collections import Counter
	res = Counter(l).get(0)
	if res == None:
		res=0
	return int(res)

df['num_zeros'] = X.map(num_zeros)

from keras.preprocessing import sequence
X = sequence.pad_sequences(X, maxlen=43) #maxlen is from trained model
y = np.ones((len(X))) #ones corresponds to Hillary

### load model and weights
from keras.models import load_model
model = load_model('log/2018-04-04-08-35model.h5')
model.load_weights('log/2018-04-04-08-35weights.h5')

scores = model.predict(X)
prob_hillary, prob_trump = np.split(scores, 2, axis=1)
df['prob_hillary'] = pd.Series(prob_hillary.reshape((prob_hillary.size)))
df['prob_trump'] = pd.Series(prob_trump.reshape((prob_trump.size)))

# scores = model.evaluate(X, y, verbose=0)
print(df)
# print('Differenting power for {} is'.format(username),scores[1])