#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:34:22 2021

@author: ml
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import numpy as np

class RFEReducer():
    _rfe_reducer = None
    
    def __init__(self, 
                 features : list,
                 labels : list, 
                 feature_names : list, 
                 model_flag : str, 
                 n_features_to_select : int,
                 seed : int = 0):
        self._features = features
        self._labels = labels
        self._feature_names = feature_names
        self._model_flag = model_flag
        self._seed = seed
        self._n_features_to_select = n_features_to_select
        
    def fit(self):
        #init model with or without seed
        model = None
        if self._seed != 0:
            if self._model_flag == "dtc":
                model = DecisionTreeClassifier(random_state = self._seed)
            elif self._model_flag == "rfc":
                model = RandomForestClassifier(random_state = self._seed)
        else:
            if self._model_flag == "dtc":
                model = DecisionTreeClassifier()
            elif self._model_flag == "rfc":
                model = RandomForestClassifier()     
        self._rfe_reducer = RFE(estimator = model, n_features_to_select = self._n_features_to_select)
        self._rfe_reducer.fit(self._features, self._labels.ravel())
        
        
    def transform(self, features : list) -> list:
        #print the names of the n best features
        output_str = "{0} best features: ".format(self._n_features_to_select)
        for i in range(0, self._n_features_to_select):
            indice  = np.where(self._rfe_reducer.ranking_ == i + 1)[0][0]
            feature_name = self._feature_names[indice]
            output_str += "\n {0}: {1} ".format(i+1, feature_name)
        print(output_str)
        reduced_features = []
        print("Before RFE:", features.shape)
        reduced_features = self._rfe_reducer.transform(features)
        print("After RFE:", reduced_features.shape)
        
        return reduced_features
    
        
        
        
    
    