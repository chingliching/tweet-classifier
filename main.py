#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 18:03:40 2017

@author: ivan

Parts of code borrowed from J. H. Wei:
https://jhwei.github.io/CMPS242_Machine_learning/docs/#/3/1
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, functools, gensim, sets, pdb, collections, tensorflow as tf, numpy as np
from tensorflow.contrib import rnn


import logging
filename='log/main_log.log'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',filename=filename, level=logging.INFO)
log=logging
#This is how you log within Spyder
##log.info('this is a test message')

import os
import tempfile
TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

tf.reset_default_graph()  #resets graph


import csv
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer

def preprocess(readfilename, writefilename):
    print("Preprocessing...")
    reader = csv.reader(open(readfilename, encoding="utf8"))
    writer = open(writefilename,'w', encoding="utf8")
    line_num = 0
    next(reader)
    labels = []
    messages=[]    
    for row in reader:
        line_num += 1 # line_num += 1 is the same as line_num++
        if line_num % 500 == 0:
            print(line_num)
        temp_label = row[0]
        temp_text = row[1]
        #get the train label list
        if temp_label == 'realDonaldTrump':
            labels.append(0)
        elif temp_label == 'HillaryClinton':
            labels.append(1)
        words = TweetTokenizer().tokenize(temp_text)
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
        words_lower = [w.lower() for w in words]
        word_num = 0
        temp_sentence = ""
        for temp_word in words_lower:
            word_num += 1
            if word_num == 1:
                temp_sentence += temp_word
            else:
                temp_sentence += " " + temp_word
        temp_sentence += "\n"
        messages.append(temp_sentence.split())
        writer.write(temp_sentence)
    writer.close()
    print("Preprocessing is done!")
    return labels, messages
    
def tokenize(readfilename, writefilename,size=50): #Tokenize with Word2Vec, use np.append
    labels, messages = preprocess(readfilename, writefilename)
    X_train_length = [len(message) for message in messages]
    max_length = max(X_train_length)
    model = gensim.models.Word2Vec(messages, min_count=1, size=size, iter=8)
    model.train(messages, total_examples=model.corpus_count, epochs=model.iter)
#Hillary is [1,0] while Trump is [0,1]
    labels=[(label==1)*[1,0]+(label==0)*[0,1] for label in labels] 
    labels=np.stack(labels,axis=0)
    
    vocab_list = []
    for word_list in messages:
        vocab_list += word_list
    count = collections.Counter(vocab_list)

    vocab_dict = dict()
    embedding_matrix = np.empty((0, size), float)

    for word in count:
        vocab_dict[word] = len(vocab_dict)
        embedding_matrix = np.vstack((embedding_matrix, model[word]))

    res=[]
    for message in messages:
        temp=[]
        for word in message:
            temp.append(vocab_dict[word])
        while len(temp)<max_length:
            temp.append(0)
        res.append(np.array(temp))
    res=np.stack(res,axis=0)

    return res, labels, X_train_length, vocab_dict, embedding_matrix

def segment(readfilename, writefilename,**kwargs): #write into data segment (database object)
    tweets,labels, X_train_length, vocab_dict, embedding_matrix = tokenize(
            readfilename,writefilename,**kwargs)
    full = sets.core.dataset.Dataset(data=tweets,target=labels)
    full.__setitem__('length',X_train_length)
    return full, vocab_dict, embedding_matrix


def split(full,num):
    """returns the num-th partition of 10-fold cross validation (0 to 9)
    full: sets.core.dataset.Dataset object
    num: int
    rtype: (Dataset, Dataset)"""
    validation_size = len(full)//10
    data=full.data.tolist()
    target=full.target.tolist()
    length=full.length.tolist()
    for _ in range(num*validation_size):
        data.insert(0,data.pop())
        target.insert(0,target.pop())
        length.insert(0,length.pop())
    data=np.array(data)
    target=np.array(target)
    length=np.array(length)
    full = sets.core.dataset.Dataset(data=data,target=target)
    full.__setitem__('length',length)
    train, validation = sets.Split(0.9)(full)
    return train, validation

def crossValidate(num_hidden, dropout, *args,training_steps=10, batch_size=93, **kwargs):
    
#    full = kwargs['full']
#    vocab_dict = kwargs['vocab_dict']
#    embedding_matrix = kwargs['embedding_matrix']
    
    tf.reset_default_graph()  #resets graph
    
#    full, vocab_dict, embedding_matrix = segment('train.csv','train_p.csv',size=embed_size)

    # Network Parameters
    num_input = 1
    time_step = full.data.shape[1]
    num_classes = 2
    
    # tf Graph input
    X = tf.placeholder(tf.int32, [None, time_step])
    X_length = tf.placeholder(tf.int32, [None])
    #embedding = tf.Variable(embedding_matrix)
    Y = tf.placeholder(tf.float16, [None, num_classes])
    
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    
    def RNN(x,x_length,weights,biases):
        """x: rank-1, x_length: rank-0, weights: rank-2, biases: rank-1
        rtype: rank-1"""
        batch_size_tmp = tf.shape(x)[0]
        embedding = tf.get_variable('embedding', [len(vocab_dict), embed_size])
        embed = [tf.nn.embedding_lookup(embedding, row)
                 for row in tf.split(x, batch_size)]
        embed = tf.reshape(embed, (batch_size_tmp, time_step, embed_size))
        embed = tf.unstack(embed, time_step, 1)
        
        lstm_cell = rnn.BasicLSTMCell(num_hidden)
        cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout)
        cell = rnn.MultiRNNCell([cell] * 1)
        outputs, states = rnn.static_rnn(
                cell, dtype=tf.float32, sequence_length=x_length, inputs=embed)
        
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
    
        index = tf.range(0, batch_size_tmp) * \
            full.data.shape[1] + tf.reshape(x_length - 1, [batch_size_tmp])
        outputs = tf.gather(tf.reshape(outputs, [-1, num_hidden]), index)
    
        return tf.matmul(outputs, weights['out']) + biases['out']
    
    logits = RNN(X, X_length, weights, biases)
    prediction = tf.nn.softmax(logits)
    tf.summary.histogram('logits', logits)
    
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss_op)
    tf.summary.scalar('loss', loss_op)
    
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
    init = tf.global_variables_initializer()
    
    merged_summary = tf.summary.merge_all()

    t2 = time.time()
    # Start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    log.info('starting training for the follwing parameters: '
    #log.info('starting training for the follwing parameters: '
             +'training_steps='+str(training_steps)
             +', batch_size='+str(batch_size)
             +', embed_size='+str(embed_size)
             +', num_hidden='+str(num_hidden)
             +', dropout='+str(dropout))
    
    fold_acc=[]
    for fold in range(10):
        log.info('starting fold '+str(fold+1)+' in 10-fold CV')
        with tf.Session(config=config) as sess:
            sess.run(init)
            train,validation = split(full,fold) 
            previous_valid_acc=0
            for step in range(1, training_steps + 1): 
                for i in range(1, train.data.shape[0] // batch_size + 1): #stochastic
                    batch=train.sample(batch_size)
                    batch_x = batch.data
                    batch_y = batch.target
                    batch_x_length = batch.length
                    batch_x_length = batch_x_length.reshape((-1))
                    summary, _ = sess.run([merged_summary, train_op], feed_dict={
                            X: batch_x, X_length: batch_x_length, Y: batch_y})
                training_loss = []
                training_acc = []
                for i in range(1, train.data.shape[0] // batch_size + 1):
                    batch=train.sample(batch_size)
                    batch_x = batch.data
                    batch_y = batch.target
                    batch_x_length = batch.length
                    batch_x_length = batch_x_length.reshape((-1))
                    loss_tmp, acc_tmp = sess.run([loss_op, accuracy], feed_dict={X: batch_x, X_length: batch_x_length,
                                                                                 Y: batch_y})
                    training_loss.append(loss_tmp)
                    training_acc.append(acc_tmp)
                log.info("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(np.mean(training_loss)) + ", Training Accuracy= " +
                      "{:.3f}".format(np.mean(training_acc)))
                validation_loss=[]
                validation_acc=[]
                for i in range(1, validation.data.shape[0]//batch_size+1):
                    batch=validation.sample(batch_size)
                    batch_x = batch.data
                    batch_y = batch.target
                    batch_x_length = batch.length
                    batch_x_length = batch_x_length.reshape((-1))
                    loss_tmp, acc_tmp = sess.run([loss_op, accuracy], feed_dict={X: batch_x, X_length: batch_x_length,
                                                                                     Y: batch_y})
                    validation_loss.append(loss_tmp)
                    validation_acc.append(acc_tmp)
                log.info("Step " + str(step) + ", num_hidden= " +
                      "{:.4f}".format(num_hidden) + ", dropout= " +
                      "{:.3f}".format(dropout)+ ", Validation Loss= " +
                      "{:.4f}".format(np.mean(validation_loss)) + ", Validation Accuracy= " +
                      "{:.3f}".format(np.mean(validation_acc)))
#                if np.mean(validation_acc)<previous_valid_acc:
#                    break
#                previous_valid_acc=np.mean(validation_acc)
#        fold_acc.append(previous_valid_acc)
        fold_acc.append(np.mean(validation_acc))
    log.info('Average accuracy is '+str(np.mean(fold_acc))+' for '
             +'training_steps='+str(training_steps)
             +', batch_size='+str(batch_size)
             +', embed_size='+str(embed_size)
             +', num_hidden='+str(num_hidden)
             +', dropout='+str(dropout))

    t3 = time.time()
    log.info('This 10-fold CV run-time: '+str(t3-t2)+' seconds')
    return np.mean(fold_acc), np.min(fold_acc)

def crossValidate_wrapper(args): #to pass multiple arguments in multiprocessing
    return crossValidate(*args)

def scan_hyperparams():
    """logs best combination of hyperparams so far"""
    result = {} #key: num_hidden and dropout, value: mean accuracy in 10-fold CV
    
    global embed_size
    global full
    global vocab_dict
    global embedding_matrix
    best_acc=0
    best_params = [0,0]
    res={}
    embed_size=20 #result from BOW_embedding
    full, vocab_dict, embedding_matrix = segment('train.csv','train_p.csv',size=embed_size)
    
#    num_hidden_range = range(20,55,5)
#    dropout_range = [.05*j for j in range(4,11)]
    num_hidden_range = range(1,6)
    dropout_range = [1] #this is actually 1-dropout
    params = [(num_hidden,dropout) for num_hidden in num_hidden_range for dropout in dropout_range]

    from multiprocessing import Pool
    agents = 5
    with Pool(processes=agents) as pool: #parallel processing
        output= pool.map(crossValidate_wrapper, params)
        
    for param, acc in zip(params,output):
        mean_acc=acc[0]
        min_acc=acc[1]
        result[param[0],round(param[1],2)]=mean_acc, mean_acc-min_acc
    log.info('results for all hyperparam combinations dict[num_hidden,dropout]=accuracy, uncer: '+str(result))
    return result


#pdb.set_trace()

#current_acc=crossValidate(num_hidden = 20,dropout=0.5)
def repeat_scan(num):
    result=[]
    for i in range(num):
        res=scan_hyperparams()
        result.append(res)
    log.info('final result of repeat_scan() is'+str(result))

def main():
    num=5
    repeat_scan(num)
    
#combined_res={}
#
#for res in result:
#    for key in res:
#        combined_res[key]=res[key]
#        
#top_res={}
#for key in combined_res:
#    if combined_res[key]>.9:
#        top_res[key]=combined_res[key]
#
#pprint(top_res)
#
#num_hidden=[]
#dropout=[]
#for key in top_res:
#    num_hidden.append(key[0])
#    dropout.append(key[1])
#print(np.mean(num_hidden))
#print(np.mean(dropout))

if __name__ == '__main__':
    t0=time.time()
    main()
    t1 = time.time()
    log.info('Code run-time: '+str(t1-t0)+' seconds')
