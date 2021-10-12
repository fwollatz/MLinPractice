#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor for removing stop words

Created on Fri Oct  8 10:24:35 2021

@author: ml
"""

from code.preprocessing.preprocessor import Preprocessor
from code.util import string_to_words_list
from nltk.corpus import stopwords

class StopWordRemover(Preprocessor):
    """removes stop words of the given input column"""
    
    def __init__(self, input_column, output_column):
        """Initialize the StopWordRemover with the given input and output column."""
        super().__init__([input_column], output_column)
    
    
    def _get_values(self, inputs : list) -> list:
        """Removing stop words from tweets"""
        sw_removed = []
        #get stop words
        stop_words = set(stopwords.words('english'))
        for tweet in inputs[0]:
            tweet_sw_removed = []
            #get words as a list
            words = string_to_words_list(tweet)
            for word in words:
                if word not in stop_words:
                    tweet_sw_removed.append(word)
            
            sw_removed.append(str(tweet_sw_removed))
        print("Removed stop words from {0} tweets!".format(len(sw_removed)))
        return sw_removed