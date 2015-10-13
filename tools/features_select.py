# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:22:30 2015

@author: Alexander
"""
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt

def select_features(features, labels, features_list):
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features, labels)
    
    #removes POI
    no_poi_features = list(features_list)
    no_poi_features.pop(0)
    sorted_scores = sorted(selector.scores_)
    new_features_list = [x for (y,x) in sorted(zip(selector.scores_,no_poi_features))]
    print "SELECTOR SCORES"
    print new_features_list, sorted_scores    
    x_axis = range(len(selector.scores_))
    x_axis_shift = [x+0.5 for x in x_axis]    
    plt.bar(x_axis, sorted_scores, 1, color="blue")
    plt.xticks(x_axis_shift, new_features_list, size='small', rotation='vertical')    
    plt.title("Scores of Features")    
    plt.axhline(y=15, color ="red")
    plt.axhline(y=10, color ="black")    
    plt.show()
    plt.clf()

    for i, feature in enumerate(features_list):
        i += 1        
        temp_features = features_list[:i]
        
        #Creates the decision tree, random forest, and SVM classifiers
        dt=tree.DecisionTreeClassifier()
        rf=RandomForestClassifier()
        svr = svm.SVC()

        #Runs GridsearchCV with all of the combinations of models and datasets above to find best model of each type
        print "Full Feature List Without PCA"
        print "SVM"
        SVM_FF=run_algorithm(svr, parametersSVM, features, labels)
        print "DT"
        DT_FF=run_algorithm(dt, parametersDT, unscaled_features, labels)
        print "RF"
        RF_FF=run_algorithm(rf, parametersRF, unscaled_features, labels)
