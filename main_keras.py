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

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from prep import vectorize

res, labels, _, vocab_dict=vectorize('tweets.csv')

ratio = .1 #proportion of test data
cutoff = int(len(res)*ratio)

X_test, X_train = res[:cutoff], res[cutoff:]
y_test, y_train = labels[:cutoff], labels[cutoff:]

max_review_length = len(res[0])
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vector_length = 20
model = Sequential()
top_words = len(vocab_dict) #number of words to keep
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
#model.add(LSTM(100,return_sequences=True))
model.add(LSTM(2))
#model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#    log.info("Accuracy: %.2f%%" % (scores[1]*100))
