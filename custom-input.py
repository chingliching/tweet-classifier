"""
20180403
Use already-trained tweet classifier on custom inputs
"""

messages = [
"Got $1.6 Billion to start Wall on Southern Border, rest will be forthcoming. Most importantly, got $700 Billion to rebuild our Military, $716 Billion next year...most ever. Had to waste money on Dem giveaways in order to take care of military pay increase and new equipment.",
"They can help solve problems with North Korea, Syria, Ukraine, ISIS, Iran and even the coming Arms Race. Bush tried to get along, but didnï¿½t have the ï¿½smarts.ï¿½ Obama and Clinton tried, but didnï¿½t have the energy or chemistry (remember RESET). PEACE THROUGH STRENGTH!",
"how about just an everyday sentence?",
"how about just an everyday sentence? how about just an everyday sentence? how about just an everyday sentence? how about just an everyday sentence?",
"what if I just put in something random?",
"anything sounds like trump",
"wow really anything sounds like trump?",
"make america great again",
"I had bagels for breakfast.",
"You didn't spend that much money this trip huh",
"Beach was not windy so it was much better than sf Beach haha",
"Kbbq was good because we intentionally did not go to an all you can eat place",
"The 626 was bigger than I expected, although we didn't get to go to night market",
"We also discovered Chinese alcohol is super gross lol",
"We watched ip man 3 and felt like we were more cultured",
"Then we played monopoly until 2am which was competitive but fun haha",
"Then we swiped tinder together for a little bit then went to sleep",
"We got into an argument about whether using tax loopholes is immoral",
"There were some instances where I needed him to remind me to brake harder"
]

import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer


df = pd.DataFrame()

messages = pd.Series(messages)

def preprocess(message):
	res = []
	for word in TweetTokenizer().tokenize(message):
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

def frac_zeros(l):
	from collections import Counter
	res = Counter(l).get(0)
	if res == None:
		res=0
	return round(res/len(l),2)

df['frac_zeros'] = X.map(frac_zeros)

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