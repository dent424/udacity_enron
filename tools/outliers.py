# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:38:33 2015

@author: Alexander
"""
from scipy.stats import mstats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

#SHOWS quantile chart of all data points
def find_outlier_vars(unscaled_features, x_labels):
    #SCALES DATA
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(unscaled_features)    
    #CONVERTS TO NUMPY ARRAY FOR QUANTILE OPERATION    
    np_features = list(features)
    np_features = np.array(np_features)
    np_unscaled = list(unscaled_features)    
    np_unscaled = np.array(np_unscaled)
    #CALCULATED RELEVANT QUANTILES
    unsc_quantiles = mstats.mquantiles(np_unscaled, prob=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99] , axis=0)
    quantiles = mstats.mquantiles(np_features, prob=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99] , axis=0)
    unsc_quantiles = zip(*unsc_quantiles)
    x_labels.pop(0)    
    for i,  q in enumerate(unsc_quantiles):
        print x_labels[i]        
        print q[0],q[1],q[2],q[3],q[4]
    labels = ['1%','5%','25%', '50%', '75%','95%','99%']
    x_axis = range(len(x_labels))     
    for i, q in enumerate(quantiles):
        plt.plot(q, label=labels[i])
    #plt.legend()
    plt.xticks(x_axis, x_labels, size='small', rotation='vertical')
    plt.xlabel("Variable Name")    
    plt.ylabel("Value")
    plt.title("Quantile Values for All Variables")    
    plt.show()
    plt.clf()

def plot_points(unscaled_features, titles):
    unsc_quantiles=zip(*unscaled_features)
    graph_names = list(titles)
    graph_names.pop(0)    
    for i, var in enumerate(unsc_quantiles):
        var = sorted(var)
        plt.plot(var, marker = 'o')
        plt.title(graph_names[i])
        plt.show()
        plt.clf()

#Data must be a dictionary of dictionaries
#key must be a list
def print_values(data, key):
    name = []
    value = []    
    for person in data:
        name.append(person)
        value.append(data[person][key])
    variables = zip(value, name)    
    variables = sorted(variables)
    for var in variables:
        print var[1], ": ", var[0]