#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of list items in the given column.

Created on Thu Oct 7 

@author: fwollatz
"""

import numpy as np
import ast
from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_URL_COUNT

# class for extracting the amount of URLs in a tweet as a feature
class NumberOfURLs(FeatureExtractor):
    
    
    def __init__(self, input_column: str):
        """
        constructor

        Parameters
        ----------
        input_column : str
            name of the input column

        Returns
        -------
        None.

        """


        super().__init__([input_column], COLUMN_URL_COUNT)
    
    # don't need to fit, so don't overwrite _set_variables()
    

    def _get_values(self, inputs: list) -> np.ndarray :
        """
        compute the url amount based on the inputs
        
        Parameters
        ----------
        inputs : list
            list of all tweets

        Returns
        -------
        result : np.ndarray
            array with the amount of urls in the tweets

        """

        #transform string to list and compute length
        list_of_lengths=[]
        for input in np.array(inputs[0]):
            urls_in_tweet=[ast.literal_eval(input)]
            list_of_lengths+=[len(urls_in_tweet[0])]

            
            
        #saving it in an array
        result = np.array(list_of_lengths)
        
        result = result.reshape(-1,1)
        return result
