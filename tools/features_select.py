# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:22:30 2015

@author: Alexander
"""
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt
from battery import battery
import numpy as np
from feature_format import featureFormat, targetFeatureSplit

#This function finds the F-scores for all variables and then iterates through them in F-score order
#Running SVM, decision tree, and random forest algorithms with GridSearchCV to find the best 
#Combination of feature-set, algorithm, and parameters
def select_features(unscaled_features, features, labels, features_list, my_dataset):
    
    #This first section finds the f-scores for all features
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features, labels)
    
    #removes POI
    no_poi_features = list(features_list)
    no_poi_features.pop(0)
    sorted_scores = sorted(selector.scores_)
    #new_features_list contains the features sorted by selector scores
    new_features_list = [x for (y,x) in sorted(zip(selector.scores_,no_poi_features))]

    #Prints a graph of features ordered by F-score    
    x_axis = range(len(selector.scores_))
    x_axis_shift = [x+0.5 for x in x_axis]    
    plt.bar(x_axis, sorted_scores, 1, color="blue")
    plt.xticks(x_axis_shift, new_features_list, size='small', rotation='vertical')    
    plt.title("Scores of Features")       
    plt.show()
    plt.clf()

    #BEGINS THE ITERATION THROUGH VARIABLES AND ALGORITHMS
    SVM = []
    DT = []
    RF = []
    
    #Loops through the feature list
    for i, feature in enumerate(no_poi_features):       
        #takes only labels and features of first ith columns        
        if i > 0 and i+1 <= len(no_poi_features):        
            new_features_list=list(new_features_list)
            j=i+1
            current_features_list = new_features_list[-j:]
            graphing_features_list = list(current_features_list)            
            current_features_list.insert(0,'poi')            
            data = featureFormat(my_dataset, current_features_list, sort_keys = True)
            labels, unscaled_features = targetFeatureSplit(data)                    
            #The battery() function runs the data through each algorithm            
            svm_output, dt_output, rf_output = battery(labels, unscaled_features, current_features_list, my_dataset)
            SVM.append(svm_output)
            print "SVM: ", SVM        
            DT.append(dt_output)
            print "DT: ", DT        
            RF.append(rf_output)
            print "RF: ", RF
        else:
            pass
    
    #Graphs each algorithm's performance with different feature sets
    x_axis = range(len(graphing_features_list))
    del x_axis[-1]
    del graphing_features_list[-1]    
    graphing_features_list.reverse()    
    print x_axis
    print graphing_features_list
    plt.xticks(x_axis, graphing_features_list, size='small', rotation='vertical')
    plt.plot(SVM)
    plt.plot(DT)
    plt.plot(RF)
    plt.title("Scores with Different Numbers of Features")       
    plt.show        