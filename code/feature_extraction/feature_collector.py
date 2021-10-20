#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collects the feature values from many different feature extractors.

Created on Wed Sep 29 12:36:01 2021

@author: lbechberger
"""
from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np



# extend FeatureExtractor for the sake of simplicity
class FeatureCollector(FeatureExtractor):
    
    
    def __init__(self, features: list):
        """
        constructor

        Parameters
        ----------
        features : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # store features
        self._features = features
        
        # collect input columns
        input_columns = []
        for feature in self._features:
            input_columns += feature.get_input_columns()
        
        # remove duplicate columns
        input_colums = list(set(input_columns))
        
        # call constructor of super class
        super().__init__(input_columns, "FeatureCollector")

    
    def fit(self, df):
        """
        overwrite fit: instead of calling _set_variables(), we forward the call to the features


        Parameters
        ----------
        df : pandas.core.frame.DataFrame
            DESCRIPTION.

        Returns
        -------
        None.

        """
        for feature in self._features:
            feature.fit(df)

    def transform(self, df) -> np.ndarray:
        """
        overwrite transform: instead of calling _get_values(), we forward the call to the features


        Parameters
        ----------
        df : pandas.core.frame.DataFrame
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        all_feature_values = []
        
        for feature in self._features:
            all_feature_values.append(feature.transform(df))
        
        result = np.concatenate(all_feature_values, axis = 1)
        return result

    def get_feature_names(self) -> list:
        """
        

        Returns
        -------
        list
            DESCRIPTION.

        """
        feature_names = []
        for feature in self._features:
            feature_name=feature.get_feature_name()
            
            #because sometimes an extractor creates more than one column, we have to check wether it is a list or a string
            if type(feature_name) == str:
                feature_names.append(feature_name)
            else:
                feature_names+=feature_name
        return feature_names