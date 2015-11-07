README

INCLUDED FILES AND FOLDERS
Project-Report.pdf - The main report for this project

FOLDER: final project
This folder contains the main files for the project including .pkl files of data and classifier
FILES: 
final_project_dataset.pkl
my_classifier.pkl
my_dataset.pkl
my_feature_list.pkl
poi_id.py 
tester.py

FOLDER: tools
This folder contains modules I've created for this project. poi_id.py will call all of these files.
FILES:
ML_algorithms.py - runs GridSearchCV and prints output. Also runs Udacity Tester
add_data.py - Adds new calculated fields poi_to_ratio and poi_from_ratio
feature_format.py - Udacity's module for formatting data in preparation for analysis
features_select.py - Returns chart of variables ranked by F-statistic and uses functions from battery.py to iteratively run models to find the best one. 
outliers.py - Makes a chart of data points to help spot outliers
battery.py - Uses the functions in ML_algorithms to run Decision tree, SVM, and Random forest algorithms
