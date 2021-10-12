#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:57:29 2021

@author: ml
"""


import ast
from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_HASHTAG_COUNT
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


        super().__init__([input_column], COLUMN_HASHTAG_COUNT)
    
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
