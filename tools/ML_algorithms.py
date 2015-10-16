# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:53:54 2015

@author: Alexander
"""

from sklearn import grid_search
from tester import test_classifier
from sklearn.pipeline import Pipeline
#import sklearn.cross_validation

#Runs selected machine learning algorithm with selected data
#Utilizes the GridSearchCV algorithm
#Returns the estimator and prints results
#Setting returned_var to "best_score" willreturn the best score instead of the estimator
def run_algorithm(model, parameters, features, labels, returned_var="estimator"):
    #folds = 10
    #cv = sklearn.cross_validation.StratifiedShuffleSplit(labels, folds, random_state = 42)    
    clf = grid_search.GridSearchCV(model, parameters, verbose=1, cv=5, scoring='f1')
    clf.fit(features, labels)
    print "BEST ESTIMATOR: ", clf.best_estimator_ 
    print "BEST SCORE: ", clf.best_score_
    print "BEST PARAMS: ", clf.best_params_
    if returned_var=="best_score":    
        return clf.best_score_
    else:
        return clf.best_estimator_
    
    
#Compiles pipeline and runs validation script provided by udacity    
def run_test(estimator, dataset, features_list):
    classifier = Pipeline(estimator)
    return test_classifier(classifier, dataset, features_list)