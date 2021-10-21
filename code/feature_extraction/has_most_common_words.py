#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:36:52 2021

@author: ml
"""

import nltk
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_CONTAINED_WORD, string_to_words_list

class HasMostCommonWords(FeatureExtractor):
    _n=5
    _n_most_common_words=None
    _suffix_words=[]
    
    
    def __init__(self, input_column,n:int=20):
        super().__init__([input_column], "#{0}_occurs".format(input_column))
        self._n=n
        
    def get_feature_name(self):
        suffixes=self._suffix_words
        names=[]
        for i in range(0,self._n):
            names+=[COLUMN_CONTAINED_WORD.format(suffixes[i])]
        return names
    
    def _set_variables(self, inputs: list):
        
        all_words = []
        for tweet in np.array(inputs[0]):
            words_list = string_to_words_list(tweet)
            all_words += words_list
            
        
        all_words_freqdist = nltk.FreqDist(all_words)
        self._n_most_common_words=all_words_freqdist.most_common(self._n)
        
        self._suffix_words=[word for (word,count) in self._n_most_common_words]
        print("Top {0} most common words are: ".format(self._n),self._suffix_words)
        
    def _get_values(self, inputs: list) -> np.ndarray :
        
        
        list_of_most_common_words=[]
        

        for tweet in inputs[0]:
            #create ohe vector of size n for n most common words
            ohe_word_used=self._n*[False]
            for i in range(self._n):
                current_word=self._n_most_common_words[i][0]
                words_in_tweet=string_to_words_list(tweet)
                if current_word in words_in_tweet:
                    ohe_word_used[i]=True
            
            #add to list
            list_of_most_common_words+=[ohe_word_used]
        
        #saving it in an array
        result = np.array(list_of_most_common_words)

        return result
        
    