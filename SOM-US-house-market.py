#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 22:57:37 2018

@author: farzam
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#convert percentage to float
def percentage_to_float(x):
    return float(x.strip('%')/100.0)

#import data
dataset=pd.read_csv('2014_Housing_Market_Analysis_Data_by_Zip_Code.csv')


columns=list(dataset.columns.values)
for i,col in enumerate(columns):
    if(i not in (0,2,12,13,28)):
        dataset[col]=dataset[col].str.rstrip('%').astype('float')/100   
    else:
        dataset[col]=dataset[col].replace( '[\$,)]','', regex=True )\
               .replace( '[(]','-',   regex=True ).astype(float)
dataset=dataset.fillna(dataset.mean())
X=dataset.iloc[:,:].values
y=dataset.iloc[:,0].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)

#train SOM
from minisom import MiniSom
som=MiniSom(4,4,input_len=30)
som.random_weights_init(X)
som.train_random(X,100)

#visualization
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
show()

#find cities
location_list=[]
mapping=som.win_map(X)
for keys in mapping.keys():
    location_list.append(sc.inverse_transform(mapping.get(keys)))
categories=[]    
for item in location_list:
    categories.append(item[:,0])





