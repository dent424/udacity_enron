# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:37:43 2015

@author: Alexander
"""
import pickle
import sys

import numpy as np

from sklearn import svm, preprocessing, tree
#from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from tester import dump_classifier_and_data

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from PCA_output import PCA_output
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
select_features(features, labels, features_list)

#REDUCED SET UP DATA (5 VARIABLES)
#Looks at the output of the select_feature() function and keeps the ones with F scores above 15 
reduced_features_list =['poi','poi_to_ratio', 'exercised_stock_options','total_stock_value',
                        'bonus','salary']
data = featureFormat(my_dataset, reduced_features_list, sort_keys = True)
reduced_labels, reduced_unscaled_features = targetFeatureSplit(data)

#Looks at the output of the select_feature() function and keeps the ones with F scores above 10
reduced_features_list2 = ['poi','poi_to_ratio', 'exercised_stock_options','total_stock_value',
                        'bonus','salary', 'deferred_income','long_term_incentive']
data = featureFormat(my_dataset, reduced_features_list2, sort_keys = True)
reduced_labels2, reduced_unscaled_features2 = targetFeatureSplit(data)

#SCALE REDUCED DATA
#Scales the data sets that have the reduced numbers of features created above. 
scaler2 = preprocessing.MinMaxScaler()
reduced_features = scaler2.fit_transform(reduced_unscaled_features)


scaler3 = preprocessing.MinMaxScaler()
reduced_features2 = scaler3.fit_transform(reduced_unscaled_features2)

#RUN PCA
#Runs PCA on full, scaled data and returns chart of explained variance for variables that are kept. 
features_pca, pca_model = PCA_output(features, n_components=5, feature_names=features_list[1:])

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

print "With PCA"
print "SVM"
SVM_PCA=run_algorithm(svr, parametersSVM, features_pca, labels)
print "DT"
DT_PCA=run_algorithm(dt, parametersDT, features_pca, labels)
print "RF"
RF_PCA=run_algorithm(rf, parametersRF, features_pca, labels)

print "Reduced Feature List Without PCA"
print "SVM"
SVM_RF=run_algorithm(svr, parametersSVM, reduced_features, reduced_labels)
print "DT"
DT_RF=run_algorithm(dt, parametersDT, reduced_unscaled_features, reduced_labels)
print "RF"
RF_RF=run_algorithm(rf, parametersRF, reduced_unscaled_features, reduced_labels)

print "Reduced Feature List 2 Without PCA"
print "SVM"
SVM_RF2=run_algorithm(svr, parametersSVM, reduced_features2, reduced_labels2)
print "DT"
DT_RF2=run_algorithm(dt, parametersDT, reduced_unscaled_features2, reduced_labels2)
print "RF"
RF_RF2=run_algorithm(rf, parametersRF, reduced_unscaled_features2, reduced_labels2)


#Set up parameters for pipeline so that the entire pipeline can be passed to grader 
scaling = preprocessing.MinMaxScaler()
pca = PCA(5)
#Sets up parameters for pipelines
estimators_SVM_FF = [('scaling', scaling), ('algorithm', SVM_FF)]
estimators_DT_FF = [('algorithm', DT_FF)]
estimators_RF_FF = [('algorithm', RF_FF)]
estimators_SVM_PCA = [('scaling', scaling),('PCA', pca), ('algorithm', SVM_PCA)]
estimators_DT_PCA = [('scaling', scaling),('PCA', pca), ('algorithm', DT_PCA)]
estimators_RF_PCA = [('scaling', scaling),('PCA', pca),('algorithm', RF_PCA)]
estimators_SVM_RF = [('scaling', scaling), ('algorithm', SVM_RF)]
estimators_DT_RF = [('algorithm', DT_RF)]
estimators_RF_RF = [('algorithm', RF_RF)]
estimators_SVM_RF2 = [('scaling', scaling), ('algorithm', SVM_RF2)]
estimators_DT_RF2 = [('algorithm', DT_RF2)]
estimators_RF_RF2 = [('algorithm', RF_RF2)]

#Run Testing with the provided grader 
print "Full SVM"
FSVM = run_test(estimators_SVM_FF, my_dataset, features_list)

print "Full DT"
FDT = run_test(estimators_DT_FF, my_dataset, features_list)

print "Full RF"
FRF = run_test(estimators_RF_FF, my_dataset, features_list)

print "PCA SVM"
PCA3SVM = run_test(estimators_SVM_PCA, my_dataset, features_list)

print "PCA DT"
PCA3DT = run_test(estimators_DT_PCA, my_dataset, features_list)

print "PCA RF"
PCA3RF = run_test(estimators_RF_PCA, my_dataset, features_list)

print "Reduced SVM"
RSVM = run_test(estimators_SVM_RF, my_dataset, reduced_features_list)

print "Reduced DT"
RDT = run_test(estimators_DT_RF, my_dataset, reduced_features_list)

print "Reduced RF"
RRF = run_test(estimators_RF_RF, my_dataset, reduced_features_list)

print "Reduced 2 SVM"
RSVM2 = run_test(estimators_SVM_RF2, my_dataset, reduced_features_list2)

print "Reduced 2 DT"
RDT2 = run_test(estimators_DT_RF2, my_dataset, reduced_features_list2)

print "Reduced 2 RF"
RRF2 = run_test(estimators_RF_RF2, my_dataset, reduced_features_list2)

#Creates an ordered chart of the F1 scores returned by the testing function provided by Udacity
f1scores = [FSVM, FDT, FRF, PCA3SVM, PCA3DT, PCA3RF, RSVM, RDT, RRF, RSVM2, RDT2, RRF2]
f1score_labels = ["Full SVM", "Full Decision Tree", "Full Random Forest", "PCA SVM", "PCA Decision Tree", "PCA Random Forest","Reduced SVM","Reduced Decision Tree","Reduced Random Forest","Reduced SVM 2","Reduced Decision Tree 2","Reduced Random Forest 2"]

points = zip(f1scores, f1score_labels)
sorted_points = sorted(points)
f1scores = [point[0] for point in sorted_points]
f1score_labels = [point[1] for point in sorted_points]

x_axis = range(len(f1scores))
plt.bar(x_axis, f1scores, 1, color="blue")
x_axis_shift = [x+0.5 for x in x_axis]            
plt.xticks(x_axis_shift, f1score_labels, size='small', rotation='vertical')
plt.ylabel('F1 Score')
plt.xlabel('Algorithm')
plt.title('F1 Scores of Different Algorithms')    
plt.show()
plt.clf()

#Pickles model, data, and selected features
dump_classifier_and_data(DT_RF, data_dict ,reduced_features_list)
