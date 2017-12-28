# Parts of code borrowed from Danijar Hafner (https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/,
#https://gist.github.com/danijar/61f9226f7ea498abce36187ddaf51ed5)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
t0 = time.time()

import functools
import tensorflow as tf
import numpy as np
import gensim
import sets
import pdb

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',filename='log/main_log.log', level=logging.INFO)
# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('log/main_log.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)
#This is how you log within Spyder
#log.info('this is a test message')


tf.reset_default_graph()  #resets graph, for experimenting

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

    def __init__(self, data, target, dropout, num_hidden=100, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def logit(self):
        # Recurrent network.
        cells = []
        for _ in range(self.num_layers):
            cell = tf.contrib.rnn.GRUCell(self.num_hidden)  # Or LSTMCell(num_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - self.dropout)
            cells.append(cell)
        network = tf.contrib.rnn.MultiRNNCell(cells)
        output, state = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        out_size = int(self.target.shape[1])
        logit = tf.contrib.layers.fully_connected(last, out_size, activation_fn=None)
       # Softmax layer.
#        weight, bias = self._weight_and_bias(
#            self._num_hidden, int(self.target.get_shape()[1]))
#        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return logit

    @lazy_property
    def prediction(self):
        prediction = tf.nn.softmax(self.logit)
        return prediction

    @lazy_property
    def cost(self):
#        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
#        return cross_entropy
        return tf.losses.softmax_cross_entropy(self.target, self.logit)

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)



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
#        print(temp_text)
        words = TweetTokenizer().tokenize(temp_text)
#        print(words)
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
#        print(temp_sentence)
        writer.write(temp_sentence)
    writer.close()
    print("Preprocessing is done!")
    return labels, messages

#def tokenize(readfilename, writefilename,size=50): #Tokenize with Word2Vec
#    res=[]
#    labels, messages = preprocess(readfilename, writefilename)
#    model = gensim.models.Word2Vec(messages, min_count=1, size=size, iter=8)
#    model.train(messages, total_examples=model.corpus_count, epochs=model.iter)
#    for message in messages:
#        res.append(np.array([np.array(model.wv[word]) for word in message]))
#    return np.array(res), labels
    
def tokenize(readfilename, writefilename,size=50): #Tokenize with Word2Vec, use np.append
    labels, messages = preprocess(readfilename, writefilename)
    max_length = max([len(message) for message in messages])
    res=[]
#    res=np.empty([0, max_length ,50], dtype=float)
    model = gensim.models.Word2Vec(messages, min_count=1, size=size, iter=8)
    model.train(messages, total_examples=model.corpus_count, epochs=model.iter)
    for message in messages:
        temp=np.empty([0, 50], dtype=float)
        for word in message:
            temp=np.append(temp,np.array([model.wv[word]]),axis=0)
        while temp.shape[0]<max_length:
            temp=np.append(temp,np.zeros((1, 50)),axis=0)
        res.append(temp)
    res=np.stack(res,axis=0)
#    labels=[[(label==1)*[1,0]+(label==0)*[0,1] for _ in range(max_length)] for label in labels] 
#Hillary is [1,0] while Trump is [0,1], this one repeats for each time step
    labels=[(label==1)*[1,0]+(label==0)*[0,1] for label in labels] 
    labels=np.stack(labels,axis=0)
    return res, labels

def segment(readfilename, writefilename): #implement data segment later
    tweets,labels = tokenize(readfilename,writefilename)
    full = sets.core.dataset.Dataset(data=tweets,target=labels)
    return full
        
def main():  #have this perform 10-fold validation
#    tweets,labels = tokenize('train.csv','train_p.csv')
#    i=0
#    valid_size=len(labels)//10
#    test_temp = np.array(tweets[valid_size*i:valid_size*(i+1)])
#    test_labels_temp = np.array(labels[valid_size*i:valid_size*(i+1)])
#    train_temp = np.array(tweets[valid_size*(i+1):])
#    labels_temp = np.array(labels[valid_size*(i+1):])
#    _, rows, row_size = train.data.shape
#    rows = num_words
#    row_size = num_train
#    num_classes = train.target.shape[1]
#    num_classes = 2
#    row_size=50
#    rows=len(labels_temp)
    full=segment('train.csv','train_p.csv')
    train, test = sets.Split(0.9)(full)
    dataset_size, max_length, input_size = train.data.shape
    num_classes = train.target.shape[1]
    data = tf.placeholder(tf.float32, [None, max_length, input_size])
    target = tf.placeholder(tf.float32, [None, num_classes])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
#    epoch=0
    for epoch in range(10):
        for _ in range(100):
            batch = train.sample(10)
            sess.run(model.optimize, {
                data: batch.data, target: batch.target, dropout: 0.5})
        error = sess.run(model.error, {
            data: test.data, target: test.target, dropout: 1})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))#    sess.run(model.optimize, {data: train.data, target: train.target, dropout: 0.5})
#    error = sess.run(model.error, {data: test.data, target: test.target, dropout: 1})
#    print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))

pdb.set_trace()

if __name__ == '__main__':
    main()
    
t1 = time.time()
print('Code run-time: ',t1-t0,'seconds')
