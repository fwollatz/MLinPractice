#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:53:52 2021

@author: ml
"""

import ast
import nltk
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_CONTAINED_EMOJI

class HasMostCommonEmojis(FeatureExtractor):

    _n=5
    _n_most_common_emojis=None
    _suffix_emojis=[]
    
    def __init__(self, input_column,n:int=20):
        super().__init__([input_column], "#{0}_occurs".format(input_column))
        self._n=n
    
    def get_feature_name(self):
        suffixes=self._suffix_emojis
        names=[]
        for i in range(0,self._n):
            names+=[COLUMN_CONTAINED_EMOJI.format(suffixes[i])]
        return names
    
    def _set_variables(self, inputs):
        
        #collect all emojis from every tweet
        all_emojis = []
        c=0
        for tweet in np.array(inputs[0]):
            emojis_in_tweet=ast.literal_eval(tweet)
            #remove empty objects
            if(len(emojis_in_tweet)!=0):
                all_emojis+=emojis_in_tweet
            else:
                c+=1
        all_emojis =[i for i in all_emojis if i != ""]
        all_emojis =[i for i in all_emojis if i != '']
        all_emojis =[i for i in all_emojis if i != "'️'"]
        all_emojis =[i for i in all_emojis if i != '""']
        all_emojis =[i for i in all_emojis if i != "'️'"]
        print("We have got "+str(c)+" tweets without emojis") 
        
        
        #compute most common emojis
        all_emojis_freqdist = nltk.FreqDist(all_emojis)
        self._n_most_common_emojis=all_emojis_freqdist.most_common(self._n)
        
        self._suffix_emojis=[emoji for (emoji,count) in self._n_most_common_emojis]
        print(self._suffix_emojis)

    def _get_values(self, inputs: list) -> np.ndarray :
        
        
        list_of_most_common_emojis=[]

        #splitting up the input
        tweets=inputs[0]

        
        for tweet in tweets:
            ohe_emojis_used=self._n*[False]
            for i in range(self._n):
                current_emojis=self._n_most_common_emojis[i][0]
                emojis_in_tweet=ast.literal_eval(tweet)
                if current_emojis in emojis_in_tweet:
                    ohe_emojis_used[i]=True
            
            #add to list
            list_of_most_common_emojis+=[ohe_emojis_used]
        
        #saving it in an array
        result = np.array(list_of_most_common_emojis)
        return result

    