#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:53:52 2021

@author: ml
"""

import ast
from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_CONTAINED_HASHTAG
import nltk
import numpy as np

class HasMostCommonHashtags(FeatureExtractor):

    _n=5
    _n_most_common_hashtags=None
    _suffix_hashtags=[]
    
    
    def __init__(self, input_column: str,n:int=20):
        super().__init__([input_column], "#{0}_occurs".format(input_column))
        self._n=n
        
    def get_feature_name(self)->list:
        """
        returns n feature names for the n most common hashtags

        Returns
        -------
        names : list
            list of all names.

        """
        suffixes=self._suffix_hashtags
        names=[]
        for i in range(0,self._n):
            names+=[COLUMN_CONTAINED_HASHTAG.format(suffixes[i])]
        return names
    
    def _set_variables(self, inputs: list):
        """
        computes the n most common hashtags
    
        Parameters
        ----------
        inputs : list
            list of all hashtags per tweet.

        Returns
        -------
        None.

        """
        
        all_hashtags = []
        for tweet in np.array(inputs[0]):
            hashtags_in_tweet=ast.literal_eval(tweet)
            all_hashtags+=hashtags_in_tweet
            
        
        all_hashtags_freqdist = nltk.FreqDist(all_hashtags)
        self._n_most_common_hashtags=all_hashtags_freqdist.most_common(self._n)
        
        self._suffix_hashtags=[hashtag for (hashtag,count) in self._n_most_common_hashtags]
        print(self._suffix_hashtags)
        
    def _get_values(self, inputs: list) -> np.ndarray :
        """
        returns a list of booleans for each tweet, corresponding to the existance of one of the n mst commonly used hashtags



        Parameters
        ----------
        inputs : list
            list of all hashtags per tweet.

        Returns
        -------
        result : np.ndarray
            ohe-list of the existance of the n different hashtags

        """
        list_of_most_common_hashtags=[]

        #splitting up the input
        tweets=inputs[0]
        

        for tweet in tweets:
            ohe_hashtag_used=self._n*[False]
            for i in range(self._n):
                current_hashtag=self._n_most_common_hashtags[i][0]
                hashtags_in_tweet=ast.literal_eval(tweet)
                if current_hashtag in hashtags_in_tweet:
                    ohe_hashtag_used[i]=True
            
            #add to list
            list_of_most_common_hashtags+=[ohe_hashtag_used]
        
        #saving it in an array
        result = np.array(list_of_most_common_hashtags)

        return result

    