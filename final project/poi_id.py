# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:37:43 2015

@author: Alexander
"""
import pickle
import sys

from sklearn import preprocessing
#from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from tester import dump_classifier_and_data

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from ML_algorithms import run_algorithm, run_test
from add_data import add_data
from features_select import select_features
from outliers import plot_points

#SET UP DATA
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
del data_dict['TOTAL']
data_dict=add_data(data_dict)
#List of features to be passed into featureFormat 
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi', 'poi_to_ratio', 'poi_from_ratio'] 

my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, unscaled_features = targetFeatureSplit(data)

#EXAMINE OUTLIERS
#Calls a function to plot graphs of data points to spot outliers
plot_points(unscaled_features, features_list)

#SCALE DATA
#Scales data for analyses which depend on comparisons of variance
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(unscaled_features)


#SEARCH TO SEE WHAT RESULTS TO KEEP
#Function returns a graph of the F-values of different variables to help determine what to keep

select_features(unscaled_features, features, labels, features_list, my_dataset)

#Feature list of highest value F-score in the select_feature function
reduced_features_list =['poi', 'exercised_stock_options','total_stock_value',
                        'bonus']
data = featureFormat(my_dataset, reduced_features_list, sort_keys = True)
reduced_labels, reduced_unscaled_features = targetFeatureSplit(data)

#SCALE REDUCED DATA
#Scales the data sets that have the reduced numbers of features created above. 
scaler2 = preprocessing.MinMaxScaler()
reduced_features = scaler2.fit_transform(reduced_unscaled_features)

#SET UP GRID SEARCH PARAMETERS AND MODELS
#Set up range of number of estimators to use in addition to previously defined ranges in random forest with GridsearchCV
n_estimators = range(10,100,10)
parametersRF = {'n_estimators': n_estimators, 'criterion':('gini','entropy')}

#Creates the decision tree, random forest, and SVM classifiers
rf=RandomForestClassifier()

#Runs GridsearchCV with the selected model and features

print "RF"
RF=run_algorithm(rf, parametersRF, reduced_unscaled_features, reduced_labels)

#Set up parameters for pipeline so that the entire pipeline can be passed to grader 
scaling = preprocessing.MinMaxScaler()

estimators_RF = [('algorithm', RF)]

print "Reduced RF"
RRF = run_test(estimators_RF, my_dataset, reduced_features_list)

#Pickles model, data, and selected features
dump_classifier_and_data(RF, data_dict ,reduced_features_list)
