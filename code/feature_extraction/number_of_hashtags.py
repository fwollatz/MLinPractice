#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of characters in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger
"""


import ast
from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import HASHTAG_COUNT
import numpy as np

# class for extracting the character-based length as a feature
class NumberOfHashtags(FeatureExtractor):

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


        super().__init__([input_column], HASHTAG_COUNT)
    
    # don't need to fit, so don't overwrite _set_variables()
    

    def _get_values(self, inputs: list) -> np.ndarray :
        """
        compute the word length based on the inputs
        
        Parameters
        ----------
        inputs : list
            list of all tweets

        Returns
        -------
        result : np.ndarray
            array with the character-length of the individual tweets

        """

        #transform string to list and compute length
        list_of_hashtags=[]
        list_of_lengths=[]
        for input in np.array(inputs[0]):
            hashtags_in_tweet=[ast.literal_eval(input)]
            list_of_lengths+=[len(hashtags_in_tweet[0])]
            list_of_hashtags+=hashtags_in_tweet
            
            
        #saving it in an array
        result = np.array(list_of_lengths)
        
        result = result.reshape(-1,1)
        return result
