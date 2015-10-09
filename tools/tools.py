# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:39:38 2015

@author: Alexander
"""

def namer(name_list, input_features, output_features):   
#namer compares two lists of numbers to determine which elements are the same.
#It then takes a list of names and returns the column names of the elements
#name_list and input_features must have the same dimensions and the names
#must be in the same numbers as the numbers in the list
    print output_features
    print input_features    
    lists = []    
    for i, o_feature in enumerate(output_features):
        temp = []        
        for i_feature in input_features:
            if i_feature == o_feature:
                temp.append(1)
            else:
                temp.append(0)
        lists.append(temp)
    print lists
    temp = []
    for i, i_feature in enumerate(input_features):
        count = 0        
        for o, o_feature in enumerate(output_features):
            count = lists[o][i] + count
        temp.append(count)
    lists = temp
    output = [a*b for a,b in zip(lists, name_list)]    
    print output            

def zero_rate(data):
    rows = len(data)
    print "Number of rows: ", rows
    columns = len(data[0])
    nans = [0]*columns
    print "Number of columns: ", columns
    for i, c in enumerate(range(columns)):
        #print "C: ", c        
        for r in range(rows):
            #print "R: ", r
            #print data[r][c]
            #print type(data[r][c])
            if math.isnan(data[r][c]):
                nans[i]=nans[i]+1
                #print "C: ", c
                #print "R: ", r
                #print data[r][c]
            else:
                pass
    print nans