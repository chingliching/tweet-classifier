#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:26:39 2018

@author: ivan
"""

import csv
from nltk.tokenize import TweetTokenizer

def combine_csv(file1,file2):
    labels1,messages1=preprocess(file1,file1[:-4]+'_p.csv')
    labels2,messages2=preprocess(file2,file2[:-4]+'_p.csv')
    combined_labels = []
    combined_messages = []
    while labels1 and labels2:
        combined_labels.append(labels1.pop(0))
        combined_labels.append(labels2.pop(0))
    while messages1 and messages2:
        combined_messages.append(messages1.pop(0))
        combined_messages.append(messages2.pop(0))
    combined_labels += labels1 + labels2
    combined_messages += messages1 + messages2
    return combined_labels, combined_messages

def combine_csv_file(file1,file2,writefilename):
    reader1 = csv.reader(open(file1,encoding='utf8'),delimiter=';')
    reader2 = csv.reader(open(file2,encoding='utf8'),delimiter=';')
    writer = open(writefilename,'w', encoding="utf8")
    for line1,line2 in zip(reader1,reader2):
        writer.write(line1[0]+';'+line1[1]+'\n')
        writer.write(line2[0]+';'+line2[1]+'\n')
    for line1 in reader1:
        writer.write(line1[0]+';'+line1[1]+'\n')
    for line2 in reader2:
        writer.write(line2[0]+';'+line2[1]+'\n')
    writer.close()

def preprocess(readfilename, writefilename,write=True):
    print("Preprocessing...")
    try:
        if write==False:
            raise FileNotFoundError
        reader = csv.reader(open(writefilename, encoding="utf8"),delimiter=';')
        labels = []
        messages=[]
        for row in reader:
            if row[0] == 'realDonaldTrump':
                labels.append(0)
            elif row[0] == 'HillaryClinton':
                labels.append(1)
            messages.append(row[1].split())
    except FileNotFoundError:
        if 'clinton' in readfilename:
            temp_label='HillaryClinton'
            tweet_loc = 4
        elif 'trump' in readfilename:
            temp_label='realDonaldTrump'
            tweet_loc = 1
        reader = csv.reader(open(readfilename,encoding='utf8'),delimiter=';')
        writer = open(writefilename,'w', encoding="utf8")
        line_num = 0
        next(reader)
        labels = []
        messages=[]
        for row in reader:
            line_num += 1 # line_num += 1 is the same as line_num++
            if line_num % 500 == 0:
                print(line_num)
            temp_text = row[tweet_loc]
            #get the train label list
            if temp_label == 'realDonaldTrump':
                labels.append(0)
            elif temp_label == 'HillaryClinton':
                labels.append(1)
            words = TweetTokenizer().tokenize(temp_text)
            for word in words:
                if 'pic.twitter.com' in word:
                    words[words.index(word)] = '<pic>'
                elif word.startswith('http'):
                    words[words.index(word)] = '<url>'
#                elif word.startswith('@'):
#                    words[words.index(word)] = '<@mention>'
#                elif word == '#': #clinton likes to include space after hashtag
#                    words[words.index(word)+1] = word+words[words.index(word)+1]
#                    words.pop(words.index(word))
                elif word[0].isdigit():
                    words[words.index(word)] = '<num>'
            if '#' in words:
                index = words.index('#')
                words[index] += words[index+1]
                words.pop(index+1)
            if '@' in words:
                index = words.index('@')
                words[index] += words[index+1]
                words.pop(index+1)
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
            writer.write(temp_label+';')
            writer.write(temp_sentence)
        writer.close()
    print("Preprocessing is done!")
    return labels, messages

def parse_file(file):
    labels,messages=preprocess('',file)
    return labels,messages

def vectorize(file):
    """turns words into indices (no embedding)"""
    from prep import parse_file
    import numpy as np
    import collections
    labels, messages = parse_file(file)
    X_train_length = [len(message) for message in messages]
    max_length = max(X_train_length)
    #Hillary is [1,0] while Trump is [0,1]
#    labels=[(label==1)*[1,0]+(label==0)*[0,1] for label in labels] 
    labels=np.stack(labels,axis=0)
    
    vocab_list = []
    for word_list in messages:
        vocab_list += word_list
    count = collections.Counter(vocab_list)

    vocab_dict = dict()
    for word in count:
        vocab_dict[word] = len(vocab_dict)+1 #zero is reserved for zero-padding

    res=[]
    for message in messages:
        temp=[]
        for word in message:
            temp.append(vocab_dict[word])
        while len(temp)<max_length:
            temp.append(0)
        res.append(np.array(temp))
    res=np.stack(res,axis=0)

    return res, labels, X_train_length, vocab_dict

def split_csv(file,d1,d2):
#    pdb.set_trace()
    """splits one csv file into three: train, test, predict;
    input decimals represent percentages, e.g. d1=.1 d2=.1 means 
    prediction set is first 10% of full set and test set is next 10%"""
    reader = csv.reader(open(file,encoding='utf8'),delimiter=';')
    next(reader) #skip header
    length = 0 #count length of file
    for line in reader:
        length +=1
    reader = csv.reader(open(file,encoding='utf8'),delimiter=';')
    writer1 = open(file[:-4]+'_predict.csv','w', encoding="utf8")
    writer2 = open(file[:-4]+'_test.csv','w', encoding="utf8")
    writer3 = open(file[:-4]+'_train.csv','w', encoding="utf8")
    i = 0
    for line in reader:
        if i < d1*length:
            i+=1
            writer1.write(line[0]+';'+line[1]+'\n')
        elif i<(d1+d2)*length:
            i+=1
            writer2.write(line[0]+';'+line[1]+'\n')
        else:
            writer3.write(line[0]+';'+line[1]+'\n')
    writer1.close()
    writer2.close()
    writer3.close()

def preprocess_sentence(message):
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
    
def clean(files):
    """ Input: list of csv file names
        Output: messages indexed and padded (np array), labels indexed (np array), vocab_dict with message indices, and handle_dict with handle indices
        Designed to be compatiable with the tweet downloader (GetOldTweets-python-master)"""
    from prep import preprocess_sentence
    messages = []
    labels = []
    handle_dict = {}
    skipped, total = 0,0
    for file in files:
        handle = file[:-4]
        handle_dict[handle] = len(handle_dict)
        handle = handle_dict[handle]
        reader = csv.reader(open(file,encoding='utf8'),delimiter=';')
        next(reader)
        for row in reader:
            total +=1
            message = row[4]
            if len(message)<10: #aka something went wrong
                skipped +=1
                print('skipping this message: {}'.format(message))
                continue
            labels.append(handle)
            message = preprocess_sentence(message)
            messages.append(message)

    import collections
    vocab_list = []
    for word_list in messages:
        vocab_list += word_list
    count = collections.Counter(vocab_list)

    vocab_dict = {'<not_in_vocab>':0} #zero is reserved for zero-padding
    for word in count:
        vocab_dict[word] = len(vocab_dict) 

    res=[]
    max_length=43
    import numpy as np
    for message in messages:
        temp=[]
        for word in message:
            if len(temp)==max_length:
                break
            temp.append(vocab_dict[word])
        while len(temp)<max_length:
            temp.append(0)
        res.append(np.array(temp))
    res=np.stack(res,axis=0)

    from keras.utils.np_utils import to_categorical 
    labels = to_categorical(labels, num_classes=len(files))

    p = np.random.permutation(len(res)) #shuffle indices around
    res, labels =  res[p], labels[p]

    print('Skipped {}/{} rows'.format(skipped, total))
    return res, labels, vocab_dict, handle_dict






    