#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:00:36 2021

@author: ml
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_WORD_COUNT, string_to_words_list
import numpy as np

#class for extracting the amount of words after general preprocessing
class NumberOfWords(FeatureExtractor):
    
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


        super().__init__([input_column], COLUMN_WORD_COUNT)
        
    def _get_values(self, inputs: list) -> np.ndarray :
        """
        compute the word count based on the inputs
        
        Parameters
        ----------
        inputs : list
            list of all tweets

        Returns
        -------
        result : np.ndarray
            array with the word-count of the individual tweets

        """

        #transform string to list and compute length
        
        
        list_of_lengths=[]
        
        for tweet in inputs[0]:
            #turn str into list of words
            list_of_words = string_to_words_list(tweet)
            if(list_of_words == ['']):
                #if tweet is empty
                list_of_lengths.append(0)
            else:
                list_of_lengths.append(len(list_of_words))
            
        #saving it in an array
        result = np.array(list_of_lengths)
        #expand dim to (tweets, 1)
        result = result.reshape(-1,1)
        return result
    