#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for all of our feature extractors.

Created on Wed Sep 29 12:22:13 2021

@author: lbechberger
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# base class for all feature extractors
#   inherits from BaseEstimator (as pretty much everything in sklearn)
#       and TransformerMixin (allowing for fit, transform, and fit_transform methods)
class FeatureExtractor(BaseEstimator,TransformerMixin):
    
    
    def __init__(self, input_columns: list, feature_name: str):
        """
        constructor

        Parameters
        ----------
        input_columns : list
            DESCRIPTION.
        feature_name : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(BaseEstimator, self).__init__()
        super(TransformerMixin, self).__init__()
        self._input_columns = input_columns
        self._feature_name = feature_name
        
    
    def get_feature_name(self)->str:
        """
        access to feature name

        Returns
        -------
        str
            DESCRIPTION.

        """
        return self._feature_name
    

    def get_input_columns(self) -> list:
        """
        access to input colums

        Returns
        -------
        list
            DESCRIPTION.

        """
        return self._input_columns
        

    
    
    def _set_variables(self, inputs: list):
        """
        # set internal variables based on input columns
        # to be implemented by subclass!

        Parameters
        ----------
        inputs : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass
    
    
    def fit(self, df):
        """
        # fit function: takes pandas DataFrame to set any internal variables

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        inputs = []
        # collect all input columns from df
        for input_col in self._input_columns:
            inputs.append(df[input_col])
        
        # call _set_variables (to be implemented by subclass)
        self._set_variables(inputs)
        
        return self
    
         
    
    def _get_values(self, inputs): 
        """
        # get feature values based on input column and internal variables
        # should return a numpy array
        # to be implemented by subclass!

        Parameters
        ----------
        inputs : ???
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print("e", type(inputs), "If you see this in your code please complete the method header of _get_values in the feature_extractor.py !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pass
        
    def transform(self, df) -> np.ndarray:
        """
        # transform function: transforms pandas DataFrame to numpy array of feature values


        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """

        inputs = []
        # collect all input columns from df
        for input_col in self._input_columns:
            inputs.append(df[input_col])
            
        result=self._get_values(inputs)
        return result