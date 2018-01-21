#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 11:19:45 2018

@author: ivan
"""

from matplotlib import pyplot as plt
import numpy as np
import matplotlib

result=[{5: 0.75316459, 10: 0.80464137, 15: 0.84409285, 20: 0.85780591, 25: 0.86371309, 30: 0.85843879, 35: 0.85147685, 40: 0.86518991, 45: 0.83839667, 50: 0.87130803}, 
         {5: 0.79240507, 10: 0.78818566, 15: 0.85274255, 20: 0.8609705, 25: 0.86624479, 30: 0.8679325, 35: 0.85316455, 40: 0.85210973, 45: 0.83059072, 50: 0.83438814}, 
         {5: 0.79936713, 10: 0.79430377, 15: 0.84282702, 20: 0.85759497, 25: 0.86392403, 30: 0.86329114, 35: 0.87130803, 40: 0.85189879, 45: 0.84493673, 50: 0.86666667}, 
         {5: 0.78945142, 10: 0.80928266, 15: 0.86181432, 20: 0.85485232, 25: 0.85590714, 30: 0.83966243, 35: 0.84978902, 40: 0.85527432, 45: 0.86160338, 50: 0.85021096}]

x=[]
y=[]
for i in range(5,55,5):
    for j in range(4):
        x.append(i)
        y.append(result[j][i])

plt.scatter(x, y, c="b", alpha=0.5)
plt.xlabel("Size of Embedded Word Vector")
plt.ylabel("Accuracy from 10-fold CV")
plt.title("Accuracy vs. Embedding Size")
plt.savefig('embed_plot.png', bbox_inches='tight',dpi=300)
plt.show()


#Calculate mean and uncer of mean for embed_size=20
mean_acc = [0.836498, 0.857806, 0.86097, 0.857595, 0.854852]
min_acc = [0.746835, 0.805907, 0.7827, 0.778481, 0.757384]
uncer = [m-n for m,n in zip(mean_acc,min_acc)]

ans = np.array(mean_acc).mean()
uncer = (1/5)*np.square(np.array(uncer)).sum()**(1/2)

print(ans,uncer) #0.8535442 0.0361210125495
