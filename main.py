# Parts of code borrowed from Danijar Hafner (https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
t0 = time.time()

import functools
import tensorflow as tf
import numpy as np

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
from nltk.corpus import stopwords
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

def tokenize_test(readfilename, writefilename,train_vocab):
    labels, messages =preprocess(readfilename, writefilename)
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(decode_error='ignore',vocabulary=train_vocab)
    X = vectorizer.fit_transform(messages)
    sms_array = X.toarray()
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(smooth_idf=False,norm='l2')
    tfidf = transformer.fit_transform(sms_array)
    return [tfidf, labels]

def main():  #have this perform 3-fold validation
    # We treat images as sequences of pixel rows.
    train_tfidf,train_labels,vocab = tokenize('train.csv','train_p.csv')
    num_train = train_tfidf.shape[0]
    num_words = train_tfidf.shape[1]
#    train, test = mnist.train, mnist.test
    
#    _, rows, row_size = train.data.shape
    rows = num_train
    row_size = num_words
#    rows = num_words
#    row_size = num_train
    print('rows:',rows,', row_size:',row_size)
#    num_classes = train.target.shape[1]
    num_classes = 2
#    data = tf.placeholder(tf.float32, [None, rows, row_size])
    data = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.float32, [None, num_classes])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    epoch=0
#    for epoch in range(10):
#        for _ in range(100):
#            batch_size = 10
#            batch_data = np.concatenate((np.array(train_tfidf.toarray()[:batch_size*_]),np.array(train_tfidf.toarray()[batch_size*(_+1):])))
#            batch_target = np.concatenate((train_labels[:batch_size*_],train_labels[batch_size*(_+1):]))
#            print('batch_data:',batch_data.shape)
#            print('batch_target:',batch_target.shape)
#            sess.run(model.optimize, {
#                data: batch_data, target: batch_target, dropout: 0.5})
#        error = sess.run(model.error, {
#            data: test.data, target: test.target, dropout: 1})
#        batch_data = np.array(train_tfidf.toarray())
#        batch_target = np.array(train_labels)
    batch_data = np.array([train_tfidf.toarray()]*model.num_hidden)
    batch_target = np.array([train_labels]*model.num_hidden)
    sess.run(model.optimize, {
            data: batch_data, target: batch_target, dropout: 0.5})
    error = sess.run(model.error, {
        data: test.data, target: test.target, dropout: 1})
    print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
    
t1 = time.time()
print('Code run-time: ',t1-t0,'seconds')


if __name__ == '__main__':
    main()
