#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:58:32 2017

@author: ivan
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',filename='log/hyperparam_scan.log', level=logging.INFO)



# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


fh = logging.FileHandler('log/hyperparam_scan.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

#This is how you log within Spyder
#log.info('this is a test message')



import os
import tempfile
TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

import time
t0 = time.time()

import numpy as np
import tensorflow as tf

import csv
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer

import gensim 

tf.reset_default_graph()  #resets graph, for experimenting



def preprocess(readfilename, writefilename):
    print("Preprocessing...")
    reader = csv.reader(open(readfilename, encoding="utf8"))
    writer = open(writefilename,'w', encoding="utf8")
    line_num = 0
    next(reader)
    labels = []         
    messages=[]
    #test_labels = []
    
    for row in reader:
#        if line_num>24: #only run through a few rows
#            return 1,2
        line_num += 1
        #print line_num
        if line_num % 500 == 0:
            print(line_num)
        temp_label = row[0]
        temp_text = row[1]
        #get the train label list
        if temp_label == 'realDonaldTrump':
            labels.append(0)
        elif temp_label == 'HillaryClinton':
            labels.append(1)
        #Make the words to lower format and Remove the stopwords
        stopWords = set(stopwords.words('english'))
#        print(temp_text)
        words = TweetTokenizer().tokenize(temp_text)
#        print(words)                
        words_lower = [w.lower() for w in words]
#        print(words_lower)
        words_lower_filter_stopwords = []
        for w in words_lower:
            if w not in stopWords:
                words_lower_filter_stopwords.append(w)
        words = words_lower_filter_stopwords
        for word in words:
            if word.startswith('http'):
                words[words.index(word)] = '<url>'
            if word.startswith('@'):
                words[words.index(word)] = '<@mention>'
            if word.startswith('#'):
                words[words.index(word)] = '<hashtag>'
            if word[0].isdigit():
                words[words.index(word)] = '<num>'
    #            if word.endswith('...'): #for some reason some characters are turned into ellipsis, e.g. 'clicks'->'cl...'
    #                words.pop(words.index(word)) 
        word_num = 0
        temp_sentence = ""
        for temp_word in words_lower_filter_stopwords:
            word_num += 1
            if word_num == 1:
                temp_sentence += temp_word
            else:
                temp_sentence += " " + temp_word
        temp_sentence += "\n"
#        if line_num>17 and line_num<24: #only run through a few rows
#            print('temp_text: ',temp_text)
#            print('words: ',words)
#            print('words_lower: ',words_lower)
#            print('temp_sentence: ',temp_sentence)
#            if line_num==23:
#                return 1,2
        messages.append(temp_sentence)
#        print(temp_sentence)
        writer.write(temp_sentence)
    writer.close()
    print("Preprocessing is done!")
    return labels, messages


    

def tokenize(readfilename, writefilename,size=25): #Tokenize with Word2Vec
    res=[]
    labels, messages = preprocess(readfilename, writefilename)
    model = gensim.models.Word2Vec(messages, min_count=1, size=size, iter=8)
    model.train(messages, total_examples=model.corpus_count, epochs=model.iter)
    for message in messages:
        res.append(sum([model.wv[word] for word in message]))
    return res, labels, None #None to account for vocab


def crossValidate(train_tfidf, train_labels, size=25, learning_rate=1, l1=5): #values of hyperparams from previous scan
    """perform 10-fold cross-validation, returns average accuracy and min accuracy"""
    accuracies=[]
    num_train = len(train_tfidf)
    valid_size = num_train//10
    for i in range(10):
        tf.reset_default_graph()
        steps=200
        tf.Session().run(tf.global_variables_initializer())
        test_temp = np.array(train_tfidf[valid_size*i:valid_size*(i+1)])
        test_labels_temp = np.array(train_labels[valid_size*i:valid_size*(i+1)])
        train_temp = np.concatenate((np.array(train_tfidf[:valid_size*i+1]),np.array(train_tfidf[valid_size*(i+1)+1:])))
        train_labels_temp = np.concatenate((train_labels[:valid_size*i+1],train_labels[valid_size*(i+1)+1:]))
        feature_columns = [tf.feature_column.numeric_column("x", shape=(size,))]
        estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                                  optimizer=tf.train.FtrlOptimizer(
                                                          learning_rate=learning_rate,
                                                          l1_regularization_strength=l1))
        input_fn = tf.estimator.inputs.numpy_input_fn(
            {"x": train_temp}, train_labels_temp, batch_size=len(train_temp), num_epochs=None, shuffle=True)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            {"x": train_temp}, train_labels_temp, batch_size=valid_size, num_epochs=steps, shuffle=False)
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            {"x": test_temp}, test_labels_temp, batch_size=len(test_temp), num_epochs=steps, shuffle=False)
        estimator.train(input_fn=input_fn, steps=steps)
#        train_metrics = estimator.evaluate(input_fn=train_input_fn)
        eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
        accuracies.append(eval_metrics['accuracy'])
        log.info('Average accuracy = '+str(np.array(accuracies).mean())+'; Min accuracy = '+str(np.array(accuracies).min()))
    return np.array(accuracies).mean()


def scanHyperparams(readfilename,writefilename):
    result = {} #key: size of embedded vector, value: mean accuracy in 10-fold CV
    all_res = []
    for size in [5*k for k in range(1,16)]:
        train_tfidf,train_labels,vocab = tokenize(readfilename,writefilename,size=size)
        mean_acc = crossValidate(train_tfidf, train_labels, size=size)
        log.info('mean accuracy = '+str(mean_acc)+' for embedding size ='+str(size))
        result[size]=mean_acc
        all_res.append([size, mean_acc])
    return [max(result.values()), max(result,key=result.get)], result


best_result, res_dict = scanHyperparams('train.csv','train_p.csv')

t1 = time.time()
print('Code run-time: ',t1-t0,'seconds')
