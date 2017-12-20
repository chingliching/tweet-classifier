#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:14:52 2017

@author: ivan
"""
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',filename='hyperparam_scan_concise.log', level=logging.INFO)



# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


fh = logging.FileHandler('hyperparam_scan_concise.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


infile = r"hyperparam_scan.log"

important = []
keep_phrases = ["Average accuracy",
              "cross-validating"]

with open(infile) as f:
    f = f.readlines()

for line in f:
    for phrase in keep_phrases:
        if phrase in line:
            important.append(line)
            break

def extractNum(string, target):
    """extracts number in string placed after target, returns float"""
    res=''
    i=string.find(target)+len(target)
    while string[i] in '1234567890.':
        res+=string[i]
        i+=1
    return float(res)

from math import *

res=[]
target1='for learning_rate= '
target2='and l1_regularization_strength= '
target3='Average accuracy = '
for i in range(len(important)):
    line=important[i]
    if i==len(important)-1 or line[30:] != important[i+1][30:]:
        if line.find(target1)>0:
            res.append([(log(extractNum(line,target1))/log(5))//1,
                    (log(extractNum(line,target2))/log(5))//1])
        elif line.find(target3)>0:
            res.append([extractNum(line,target3)])


res_c = []
for i in range(len(res)):
    if i==len(res)-1 or len(res[i])!=len(res[i+1]):
        res_c.append(res[i])
        


params=[]
acc=[]
for i in range(len(res_c)):
    if i%2==0:
        params.append(res_c[i])
    elif i%2==1:
        acc.append(res_c[i])
params.pop()



learning_rate = [param[0] for param in params]
l1 = [param[1] for param in params]
accu = [accuracy[0] for accuracy in acc]



acc_dict={}
for i in range(len(learning_rate)//2):
    acc_dict[learning_rate[i],l1[i]]=accu[i]

