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
from keras.layers import Dense, BatchNormalization, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.regularizers import l1_l2

from prep import clean

files = ['hillaryclinton.csv',
        'realdonaldtrump.csv',
        'jimmyfallon.csv',
        'barackobama.csv',
        'conanobrien.csv']

res, labels, vocab_dict, handle_dict =clean(files)

ratio = .0 #proportion of test data
cutoff = int(len(res)*ratio)

X_test, X_train = res[:cutoff], res[cutoff:]
y_test, y_train = labels[:cutoff], labels[cutoff:]

max_review_length = len(res[0])
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vector_length = 20
rs = [100] #num units in each recurrent layer
ds = [100, len(handle_dict)] #last value is number of classes
k_l1 = 0
k_l2 = .1
b_l1 = 0
b_l2 = .1
dropout_rate=[0.1, 0.1]

def train():
    model = Sequential()
    top_words = len(vocab_dict)+1 #number of words to keep (account for reserving zero for zero-padding)
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    #model.add(LSTM(100,return_sequences=True))
    for r in rs[:-1]:
        model.add(LSTM(r,return_sequences=True))
    # model.add(Dropout(dropout_rate[0]))
    # model.add(BatchNormalization())
    model.add(LSTM(rs[-1]))
    model.add(Dropout(dropout_rate[1]))
    for d in ds[:-1]:
        model.add(Dense(d, activation='sigmoid'))
    model.add(Dense(ds[-1], activation='softmax', kernel_regularizer=l1_l2(k_l1,k_l2), bias_regularizer=l1_l2(b_l1,b_l2)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy','categorical_crossentropy'])
    print(model.summary())
    # history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=1)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=64, shuffle=True, verbose=1)

    # Final evaluation of the model
    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1]*100))
    # first="Accuracy: %.2f%%" % (scores[1]*100)
    # second=' for recurrent layers '+str(rs) + ' and dense layers '+str(ds)
    # log.info(first+second)
    
    # writer = open('results.csv','w', encoding="utf8")
    # writer.write(str(rs)+',')
    # writer.write(str(ds)+',')
    # writer.write(str(round(scores[1],4))+'\n')
    # writer.close()
    
    return model, history

second='''{}-class classification: recurrent layers {} and dense layers {}
 and regularization hyperparams {}, {}, {}, {}
 and dropout hyperparams {}
 '''.format(len(handle_dict),str(rs),str(ds),k_l1,k_l2,b_l1,b_l2,dropout_rate)

log.info(second)

t0=time.time()

num_trials = 1
res=[]
for i in range(num_trials):
    model, history=train()
    log.info(str(history.history))
# for i in range(len(res)):
#     res[i] = round(res[i],4)


# generate run id
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
with open('log/{}vocab_dict.json'.format(run_id), 'w') as f:
    json.dump(vocab_dict, f)



log.info('final_result '+':'+str(res)+str(model.metrics))
t1 = time.time()
log.info('Code run-time: '+str(t1-t0)+' seconds')

