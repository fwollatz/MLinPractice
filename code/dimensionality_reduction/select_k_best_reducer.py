#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:38:43 2021

this class represents the SelectKBest using mutal info dim. reducer

@author: ml
"""

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from numpy import ndarray

class SelectKBestReducer():
    _select_k_best = None
    
    def __init__(self, features : list, labels : list):
        self._features = features
        self._labels = labels
        
    
    def fit(self, k : int):
        self._select_k_best = SelectKBest(mutual_info_classif, k = k)
        self._select_k_best.fit(self._features, self._labels.ravel())
       
    def transform(self, features : list) -> list:
        reduced_features = []
        reduced_features = self._select_k_best.transform(features)
        return reduced_features
        
        
    # resulting feature names based on support given by SelectKBest
    def get_feature_names(self, names : list) -> list:
        """
        Select k best features and returns a list of the feature names

        """
        support = self._select_k_best.get_support()
        result = []
        for name, selected in zip(names, support):
            if selected:
                result.append(name)
        return result
    
    def get_scores(self) -> ndarray:
        return self._select_k_best.scores_
        