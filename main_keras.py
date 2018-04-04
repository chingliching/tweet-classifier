#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:46:32 2018

@author: ivan
"""
import logging
filename='log/main_keras.log'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',filename=filename, level=logging.INFO)
log=logging
import time

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from prep import vectorize

res, labels, _, vocab_dict=vectorize('tweets.csv')

temp = []
for label in labels:
    if label==0:
        temp.append([0,1])
    elif label==1:
        temp.append([1,0])
    else:
        temp.append(None)
labels = np.array(temp)


ratio = .2 #proportion of test data
cutoff = int(len(res)*ratio)

X_test, X_train = res[:cutoff], res[cutoff:]
y_test, y_train = labels[:cutoff], labels[cutoff:]

max_review_length = len(res[0])
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vector_length = 20
rs = [5] #num units in each recurrent layer
ds = [1] #last value must be 1

def train():
    model = Sequential()
    top_words = len(vocab_dict)+1 #number of words to keep (account for reserving zero for zero-padding)
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    #model.add(LSTM(100,return_sequences=True))
    for r in rs[:-1]:
        model.add(LSTM(r,return_sequences=True))
    model.add(LSTM(rs[-1]))
    for d in ds:
        model.add(Dense(d, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    first="Accuracy: %.2f%%" % (scores[1]*100)
    second=' for recurrent layers '+str(rs) + ' and dense layers '+str(ds)
    log.info(first+second)
    
    writer = open('results.csv','w', encoding="utf8")
    writer.write(str(rs)+',')
    writer.write(str(ds)+',')
    writer.write(str(round(scores[1],4))+'\n')
    writer.close()
    
    return scores[1], model, history

second=' for recurrent layers '+str(rs) + ' and dense layers '+str(ds)
t0=time.time()

num_trials = 1
res=[]
for i in range(num_trials):
    acc, model, history=train()
    res.append(round(acc,3))
# for i in range(len(res)):
#     res[i] = round(res[i],4)

#generate run id
from time import gmtime, strftime
run_id = strftime("%Y-%m-%d-%H-%M", gmtime())
#save model
model.save('log/{}model.h5'.format(run_id))  # creates a HDF5 file
#save weights
model.save_weights('log/{}weights.h5'.format(run_id))
#save history
with open('log/{}history.txt'.format(run_id), 'w') as file:
     file.write(str(history.history))
#save vocab_dict
import json
with open('log/vocab_dict.json', 'w') as f:
    json.dump(vocab_dict, f)



log.info('final_result for '+second+':'+str(res))
t1 = time.time()
log.info('Code run-time: '+str(t1-t0)+' seconds')