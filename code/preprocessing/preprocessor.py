#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Superclass for all preprocessors.

Created on Tue Sep 28 17:06:35 2021

@author: lbechberger
"""

from pandas.core.frame import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

# inherits from BaseEstimator (as pretty much everything in sklearn)
#   and TransformerMixin (allowing for fit, transform, and fit_transform methods)
class Preprocessor(BaseEstimator,TransformerMixin):
    
    def __init__(self, input_columns : list, output_column : str):
        """"
        Constructor: initializes preprocessor object
        """
        super(BaseEstimator, self).__init__()
        super(TransformerMixin, self).__init__()
        self._output_column = output_column
        self._input_columns = input_columns
    
    
    def _set_variables(self, inputs):
        """
        set internal variables based on input columns
        to be implemented by subclass!
        """
        print("Type inputs: " + str(type(inputs)))
        pass
    
    def fit(self, df : DataFrame):
        """
        fit function: takes pandas DataFrame to set any internal variables
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
        get preprocessed column based on the inputs from the DataFrame and internal variables
        to be implemented by subclass!
        """
        print("Type inputs: " + str(type(inputs)))
        pass
        
    def transform(self, df : DataFrame) -> DataFrame:
        """
        transform function: transforms pandas DataFrame based on any internal variables
        """
        inputs = []
        # collect all input columns from df
        for input_col in self._input_columns:
            inputs.append(df[input_col])
        
        # add to copy of DataFrame
        df_copy = df.copy()
        df_copy[self._output_column] = self._get_values(inputs) 
        return df_copy