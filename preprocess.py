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
    writer.close

def preprocess(readfilename, writefilename):
    print("Preprocessing...")
    try:
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
#            temp_label = row[0]
            temp_label = 'HillaryClinton'
            temp_text = row[4]
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