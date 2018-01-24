#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:36:51 2018

@author: ivan
"""
import numpy as np

infile = r"main_log.log"

important = []

with open(infile) as f:
    f = f.readlines()

start=False
for line in f:
    if '2018-01-24 01:53:52,076' in line: #beginning of the relevant run
        start=True
    if start:
        if "Step 10" in line and "Validation" in line:
            important.append(line)

def extractNum(string, target):
    """extracts number in string placed after target, returns float"""
    res=''
    i=string.find(target)+len(target)
    while string[i] in '1234567890.':
        res+=string[i]
        i+=1
    return float(res)

res=[]
for line in important:
    val=extractNum(line,'Validation Accuracy= ')
    res.append(val)
            
mean = np.mean(np.array(res))
uncer = np.std(np.array(res))
print(mean,'+/-',2*uncer)