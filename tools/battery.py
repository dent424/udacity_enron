# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:01:32 2015

@author: Alexander
"""
from ML_algorithms import run_algorithm, run_test 
import numpy as np
from sklearn import svm, preprocessing, tree
from sklearn.ensemble import RandomForestClassifier

#This function runs data through GridSearchCV with SVM, Decision Tree, and Random Forest
#Returns the F1 score for each of these algorithms according to the Udacity Tester
def battery(labels, unscaled_features, features_list, my_data):
    #SET UP GRID SEARCH PARAMETERS AND MODELS
    #Sets up ranges of C, Gamma, and Kernel to test with SVM in GridsearchCV
    scaler = preprocessing.MinMaxScaler()
    scaled_features = scaler.fit_transform(unscaled_features)    
    
    C_range = np.logspace(-2,5,8)
    gamma_range = np.logspace(-5,2,8)
    parametersSVM = {'kernel':('linear','rbf'),'C':C_range, 'gamma':gamma_range}
    #Sets up range of max depth, min sample split, and criterion to test with GridsearchCV
    max_depth = range(2,20,2)
    min_samples_split = range(2,10,2)
    parametersDT ={'criterion':('gini','entropy'),'max_depth':max_depth, 'min_samples_split':min_samples_split}
    #Set up range of number of estimators to use in addition to previously defined ranges in random forest with GridsearchCV
    n_estimators = range(10,100,10)
    parametersRF = {'n_estimators': n_estimators, 'criterion':('gini','entropy')}
    
    print "CURRENT FEATURES: ", features_list 
    dt=tree.DecisionTreeClassifier()
    rf=RandomForestClassifier()
    svr = svm.SVC()
    
    print "SVM"
    SVM=run_algorithm(svr, parametersSVM, scaled_features, labels)
    print "DT"
    DT=run_algorithm(dt, parametersDT, unscaled_features, labels)
    print "RF"
    RF=run_algorithm(rf, parametersRF, unscaled_features, labels)
    
    scaling = preprocessing.MinMaxScaler()    
    estimators_SVM = [('scaling', scaling), ('algorithm', SVM)]
    estimators_DT = [('algorithm', DT)]
    estimators_RF = [('algorithm', RF)]
    
    print "CURRENT FEATURES: ", features_list    
    print type(my_data)
    SVM_score = run_test(estimators_SVM, my_data, features_list)
    DT_score = run_test(estimators_DT, my_data, features_list)
    RF_score = run_test(estimators_RF, my_data, features_list)    
    
    return SVM_score, DT_score, RF_score