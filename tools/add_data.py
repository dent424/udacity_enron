# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:37:11 2015

@author: Alexander
"""
import math

#Adds new variables to data dictionary
def add_data(data_dict):
    for point in data_dict:
        if (math.isnan(float(data_dict[point]['from_messages'])) or math.isnan(float(data_dict[point]['from_this_person_to_poi']))):                   
            data_dict[point]['poi_to_ratio'] = 0            
        else:
            data_dict[point]['poi_to_ratio']=float(data_dict[point]['from_this_person_to_poi'])/float(data_dict[point]['from_messages'])             
        for point in data_dict:
            if (math.isnan(float(data_dict[point]['to_messages'])) or math.isnan(float(data_dict[point]['from_poi_to_this_person']))):                   
                data_dict[point]['poi_from_ratio'] = 0            
            else:
                data_dict[point]['poi_from_ratio']=float(data_dict[point]['from_poi_to_this_person'])/float(data_dict[point]['to_messages'])             
    return data_dict