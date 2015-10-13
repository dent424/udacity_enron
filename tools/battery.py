# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:01:32 2015

@author: Alexander
"""
from ML_algorithms import run_algorithm 
import numpy as np
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier

def run_battery(labels, unscaled_features, scaled_features, data):
    #SET UP GRID SEARCH PARAMETERS AND MODELS
    #Sets up ranges of C, Gamma, and Kernel to test with SVM in GridsearchCV
    C_range = np.logspace(-2,5,8)
    gamma_range = np.logspace(-5,2,8)
    parametersSVM = {'kernel':('linear','rbf'),'C':C_range, 'gamma':gamma_range}
    #Sets up range of max depth, min sample split, and criterion to test with GridsearchCV
    max_depth = range(2,20,2)
    min_samples_split = range(2,10,2)
    parametersDT ={'criterion':('gini','entropy'),'max_depth':max_depth, 'min_samples_split':min_samples_split}
    #Set up range of number of estimators to use in addition to previously defined ranges in random forest with GridsearchCV
    n_estimators = range(10,100,10)
    parametersRF = {'n_estimators': n_estimators, 'criterion':('gini','entropy'), 'max_depth':max_depth}
    
    dt=tree.DecisionTreeClassifier()
    rf=RandomForestClassifier()
    svr = svm.SVC()
    
    print "DT"
    SVM=run_algorithm(svr, parametersSVM, scaled_features, labels)
    print "DT"
    DT=run_algorithm(dt, parametersDT, unscaled_features, labels)
    print "RF"
    RF=run_algorithm(rf, parametersRF, unscaled_features, labels)
    
    return SVM, DT, RF