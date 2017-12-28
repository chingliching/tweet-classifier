#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:10:09 2017

@author: ivan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',filename='log/find_words_scan.log', level=logging.INFO)


# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


fh = logging.FileHandler('log/find_words_scan.log')
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
import pdb

import csv
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer

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
        words = TweetTokenizer().tokenize(temp_text)
        words_lower = [w.lower() for w in words]
        words_lower_filter_stopwords = []
        for w in words_lower:
            if w not in stopWords:
                words_lower_filter_stopwords.append(w)
        words = words_lower_filter_stopwords
        for word in words:  #looking too much into these information will presumably lead to overfitting
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
        messages.append(temp_sentence)
#        print(temp_sentence)
        writer.write(temp_sentence)
    writer.close()
    print("Preprocessing is done!")
    return labels, messages


#tokenize the texts and apply tf-idf transform
def tokenize(readfilename, writefilename):
    labels, messages = preprocess(readfilename, writefilename)
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df=3,decode_error='ignore')
    X = vectorizer.fit_transform(messages)
    sms_array = X.toarray()
    vocab = vectorizer.vocabulary_
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(smooth_idf=False, norm='l2')
    tfidf = transformer.fit_transform(sms_array)
    return [tfidf, labels, vocab]

def oneHot(vocab):
    """return tfidf of one-hot vectors for each word in vocab"""
    oneHotMessages=[]
    for word in vocab:
        oneHotMessages.append(word)
    messages=oneHotMessages
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df=1,decode_error='ignore')
    X = vectorizer.fit_transform(messages)
    sms_array = X.toarray()
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(smooth_idf=False, norm='l2')
    tfidf = transformer.fit_transform(sms_array)
    return tfidf


train_tfidf,train_labels,vocab = tokenize('train.csv','train_p.csv')
num_train = train_tfidf.shape[0]
num_words = train_tfidf.shape[1]


feature_columns = [tf.feature_column.numeric_column("x", shape=[num_words])]


def crossValidate(train_tfidf,train_labels):
    """perform 10-fold cross-validation"""
    valid_size = num_train
    result = {}
    mean_acc_dict={}
    min_acc_dict={}
#    for learning_rate in [5**j for j in range(0,5)]:
#        for l1 in [5**k for k in range(0,5)]:
    learning_rate=1
    l1=5
#            log.info('now cross-validating for learning_rate= '+str(learning_rate)+'and l1_regularization_strength= '+str(l1))
    t0 = time.time()
#    accuracies=[]
#    for i in range(10):
    tf.reset_default_graph()
    steps=200
    tf.Session().run(tf.global_variables_initializer())
#    test_temp = np.array(train_tfidf.toarray()[valid_size*i:valid_size*(i+1)])
#    test_labels_temp = np.array(train_labels[valid_size*i:valid_size*(i+1)])
#    train_temp = np.concatenate((np.array(train_tfidf.toarray()[:valid_size*i]),np.array(train_tfidf.toarray()[valid_size*(i+1):])))
#    train_labels_temp = np.concatenate((train_labels[:valid_size*i],train_labels[valid_size*(i+1):]))
    train_temp=np.array(train_tfidf.toarray())
    train_labels_temp=np.array(train_labels)
    oneHotMessages=oneHot(vocab)
    test_temp=np.array(oneHotMessages.toarray())
    estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                              optimizer=tf.train.FtrlOptimizer(
                                                      learning_rate=learning_rate,
                                                      l1_regularization_strength=l1))
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": train_temp}, train_labels_temp, batch_size=valid_size, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": train_temp}, train_labels_temp, batch_size=valid_size, num_epochs=steps, shuffle=False)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            {"x": test_temp}, batch_size=valid_size,num_epochs=1,shuffle=False)
#    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#        {"x": test_temp}, test_labels_temp, batch_size=num_train-valid_size, num_epochs=steps, shuffle=False)
    estimator.train(input_fn=input_fn, steps=steps)
#                train_metrics = estimator.evaluate(input_fn=train_input_fn)
#    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
#    accuracies.append(eval_metrics['accuracy'])
    predict_results = estimator.predict(input_fn=predict_input_fn)
    for predict_result, vocab_word in zip(predict_results,vocab):
        result[vocab_word]=predict_result['probabilities']
    t1 = time.time()
#    log.info('run-time for these hyperparams is '+str(t1-t0)+' seconds')
#    log.info('Average accuracy = '+str(np.array(accuracies).mean())+'; Min accuracy = '+str(np.array(accuracies).min()))
#    mean_acc_dict[learning_rate,l1]=np.array(accuracies).mean()
#    min_acc_dict[learning_rate,l1]=np.array(accuracies).min()
#    log.info('Intermediate result: max mean accuracy is '+str(max(mean_acc_dict.values()))+' for (learning_rate, l1) = '+str(max(mean_acc_dict,key=mean_acc_dict.get)))
#    log.info('Intermediate result: best min accuracy is '+str(max(min_acc_dict.values()))+' for (learning_rate, l1) = '+str(max(min_acc_dict,key=min_acc_dict.get)))
#
#    log.info('Result: max average accuracy is for (learning_rate, l1) = '+str(max(mean_acc_dict,key=mean_acc_dict.get))+
#    'and highest value of min accuracy is for (learning_rate, l1) = '+str(max(min_acc_dict,key=min_acc_dict.get)))
#        print(train_metrics,eval_metrics)
#        result.append([train_metrics,eval_metrics])
    return result

pdb.set_trace()

result = crossValidate(train_tfidf,train_labels)

def five_words(result):
    """returns the five most distinct words for each presidential candidate"""
    dictionary={}
    for word in result:
        dictionary[word]=result[word][0]
    sorted_dict = sorted(dictionary.iteritems(), key=operator.itemgetter(1), reverse=True)
    highest5 = sorted_dict[:5].values()
    lowest5 = sorted_dict[-5:-1].values()
    return [highest5, lowest5]

pdb.set_trace()

distinct_words = five_words(result)
    
t1 = time.time()
print('Code run-time: ',t1-t0,'seconds')
