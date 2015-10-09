# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:11:52 2015

@author: Alexander
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def PCA_output(features, n_components=None, num_components=0, feature_names=None):
    #RUNS PCA on features, graphs variance of components returning transformed data and model
    #n_components is n_components parameter in PCA
    #num_components is number of components to graph    
    #RUNS PCA and transforms features    
    pca = PCA(n_components).fit(features)
    features_pca = pca.transform(features)    
    explained_variance_ratio = pca.explained_variance_ratio_
    pca_components = pca.components_
    
    #PLOTS VARIANCE OF PCA COMPONENTS    
    x_axis = range(len(explained_variance_ratio))
    plt.bar(x_axis, explained_variance_ratio, 1, color="blue")
    plt.axhline(y=0.05, color ="red")
    plt.ylabel('% Variance Explained')
    plt.xlabel('Component Number')
    plt.title('Variance Explained by Components')    
    plt.show()
    plt.clf()
    
    #PLOTS COMPONENTS 
    if num_components!=0:
        i=0        
        while i < num_components:        
            i += 1            
            x_axis = range(len(pca_components[i]))  
            plt.bar(x_axis, pca_components[i], 1, color="blue")
            if feature_names!=None:
                x_axis_shift = [x+0.5 for x in x_axis]            
                plt.xticks(x_axis_shift, feature_names, size='small', rotation='vertical')
            plot_name = "Component %i" % (i)
            plt.title(plot_name)  
            plt.show()            
            plt.clf()        
        
    return features_pca, pca
    
