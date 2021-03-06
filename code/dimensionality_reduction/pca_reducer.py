#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:41:12 2021

@author: ml
"""

from code.util import PCA_EXPLAINED_VARIANCE_THRESHOLD
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np


class PCAReducer():
    _pca_reducer = None
    
    def __init__(self, features : list, labels : list, seed: int = 0, use_normalizer : bool = True):
        self._features = features
        self._labels = labels
        self._seed = seed
        self._use_normalizer = use_normalizer
        
    def fit(self):
        """
        init PCA Object with and without seed
        adds normalizer/standardizer
        Returns
        -------
        None.

        """
        
        if self._seed != 0:
            self._pca_reducer = PCA(random_state=self._seed)
        else:
            self._pca_reducer = PCA()
            
        #create small pipeline for standardizing/normalizing before PCA
        if self._use_normalizer:
            self._pipeline = Pipeline([('scaling', Normalizer()), ('pca', self._pca_reducer)])
        else:
            self._pipeline = Pipeline([('scaling', StandardScaler()), ('pca', self._pca_reducer)])
        
        self._pipeline.fit(self._features)
        
        
    def transform(self, features : list) -> list:
        """
        Computes the cumulative explained variance ratio and returns the best features

        Parameters
        ----------
        features : list
            All features.

        Returns
        -------
        list
            the best transformed features.

        """
        reduced_features = []
        
        pc_counter = 0
        cumulative_explained_variance = 0
        
        #sort components and count how many components are needed to surpass the threshold
        sorted_explained_variance_ratio = np.sort(self._pca_reducer.explained_variance_ratio_)[::-1]
        for ratio in sorted_explained_variance_ratio:
            pc_counter += 1
            cumulative_explained_variance += ratio
            if(cumulative_explained_variance > PCA_EXPLAINED_VARIANCE_THRESHOLD):
                break
        
        transformed_all = self._pipeline.transform(features)
        print("Before PCA:",features.shape)
        #take only the top principle components
        reduced_features = transformed_all[:,0:pc_counter]
        print("After PCA: ", reduced_features.shape)
        return reduced_features