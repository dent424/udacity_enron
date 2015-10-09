# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:38:33 2015

@author: Alexander
"""
import matplotlib.pyplot as plt

#SHOWS each variable with points plotted according to rank.
#These plots help spot outliers
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