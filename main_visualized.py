#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20180119

@author: ivan

Parts of code borrowed from J. H. Wei:
https://jhwei.github.io/CMPS242_Machine_learning/docs/#/3/1
"""

#import main

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, functools, gensim, sets, pdb, collections, tensorflow as tf, numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

t0 = time.time()

import logging
filename='log/main_visualized_log.log'
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
    vocab_by_value={}
    for key in vocab_dict:
        vocab_by_value[vocab_dict[key]]=key

    return res, labels, X_train_length, vocab_dict, embedding_matrix, vocab_by_value

def segment(readfilename, writefilename,**kwargs): #write into data segment (database object)
    tweets,labels, X_train_length, vocab_dict, embedding_matrix, vocab_by_value = tokenize(
            readfilename,writefilename,**kwargs)
    full = sets.core.dataset.Dataset(data=tweets,target=labels)
    full.__setitem__('length',X_train_length)
    return full, vocab_dict, embedding_matrix, vocab_by_value


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

def train_and_visualize(num_hidden, dropout, *args,training_steps=10, batch_size=93, **kwargs):
    
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
        
#        pdb.set_trace()

        outputs, states = rnn.static_rnn(
                cell, dtype=tf.float32, sequence_length=x_length, inputs=embed)
        
#        print(states)
        
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
    
        index = tf.range(0, batch_size_tmp) * \
            full.data.shape[1] + tf.reshape(x_length - 1, [batch_size_tmp])
        outputs = tf.gather(tf.reshape(outputs, [-1, num_hidden]), index)
    
        return tf.matmul(outputs, weights['out']) + biases['out'], states, embed
    
    logits, states, embed = RNN(X, X_length, weights, biases)
    prediction = tf.nn.softmax(logits)
#    tf.summary.histogram('logits', logits)
    
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss_op)
#    tf.summary.scalar('loss', loss_op)
    
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#    tf.summary.scalar('accuracy', accuracy)
    
    init = tf.global_variables_initializer()
    
#    merged_summary = tf.summary.merge_all()
#    states_summary=tf.summary.tensor_summary(states_node ,states)

#    # Start training
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
    
    fold=5 #choose validation batch
#    with tf.Session(config=config) as sess:
    with tf.Session() as sess:
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
#                pdb.set_trace()
#                summary, _,states = sess.run([merged_summary, train_op, states], feed_dict={
#                        X: batch_x, X_length: batch_x_length, Y: batch_y})
#                pdb.set_trace()
                _, state = sess.run([train_op, states], feed_dict={
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
            if np.mean(validation_acc)<previous_valid_acc:
                break
            previous_valid_acc=np.mean(validation_acc)
            
            
        #visualize activation of each neuron at each word in 20 tweets    
        dropout=0
        result=[] 
        num_tweets=5 #number of tweets to observe
        batch=validation.sample(num_tweets) #change back to batch_size as needed
        batch_x = batch.data
        batch_y = batch.target
        batch_x_length = batch.length
        for i in range(num_tweets):
            batch_x_input = []
            batch_y_input = np.array([batch_y[i]]*93)
            batch_x_length_input = []
            for j in range(1,1+batch_x.shape[1]):
                batch_x_input.append(np.append(batch_x[i][:j], [0]*(batch_x.shape[1]-j)))
                batch_x_length_input.append(j)
            for j in range(batch_x.shape[1]+1,93+1):
                batch_x_input.append(np.array([0]*batch_x.shape[1]))
                batch_x_length_input.append(0)
            batch_x_input = np.array(batch_x_input)
            batch_x_length_input = np.array(batch_x_length_input).reshape((-1))
#            state_list, embed_list = sess.run([states, embed], feed_dict={
#                        X: batch_x_input, X_length: batch_x_length_input, Y: batch_y_input})
#            pdb.set_trace()
            state_list, predicted = sess.run([states, prediction], feed_dict={
                        X: batch_x_input, X_length: batch_x_length_input, Y: batch_y_input})
#            loss_tmp, acc_tmp = sess.run([loss_op, accuracy], feed_dict={X: batch_x, X_length: batch_x_length,
#                                                                                 Y: batch_y})
#            print(acc_tmp)
#            pdb.set_trace()
            res={} # key: three keys for each word in tweet
            for j in range(batch_x_length[i]):
                res[j,'state'] = state_list[0][0][j]
                res[j,'word'] = vocab_by_value[batch_x[i][j]]
                res[j,'predicted']=predicted[j]
            result.append(res)

    return previous_valid_acc, result, batch_y


embed_size=20
full, vocab_dict, embedding_matrix, vocab_by_value = segment('train.csv','train_p.csv',size=embed_size)
num_hidden=20
dropout= .9 #this is actually 1-dropout...

previous_valid_acc, result, batch_y = train_and_visualize(num_hidden, dropout)

def plot_neuron(neuron_num):    
    for i in range(len(result)): #each res is a tweet
        words=[] #contains words and prediction so far
        acts=[]
        res=result[i]
#        pdb.set_trace()
        for word_num in range(len(res)//3):
            acts.append(res[word_num,'state'][neuron_num]) #activations of a single neuron at each word
            pClinton = '{:.2f}'.format(res[word_num,'predicted'][0])
            pTrump = '{:.2f}'.format(res[word_num,'predicted'][1])
            words.append(res[word_num,'word']+' (' + pClinton +', ' + pTrump + ')')
        plt.rcdefaults()
        fig, ax = plt.subplots()
        y_pos = np.arange(len(words))
        ax.barh(words, acts)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Activation')
        ax.set_ylabel('Word and Predictions(Clinton, Trump) (True Value: '+str(batch_y[i])+')')
        ax.set_title('Neuron #'+str(neuron_num)+'(Overall accuracy: '+'{:.2f}'.format(previous_valid_acc)+')')
        plt.xlim(-1,1)
        plt.tight_layout()
        plt.savefig('activations_visualized/run4/neuron'+str(neuron_num)+'tweet'+str(i))

            
for neuron_num in range(num_hidden):
    plot_neuron(neuron_num)    
        
#plot_neuron(0)
    


t1 = time.time()
log.info('Code run-time: '+str(t1-t0)+' seconds')
