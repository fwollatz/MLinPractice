#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:53:52 2021

@author: ml
"""

import ast
from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_CONTAINED_EMOJI
import nltk
import numpy as np
import re

class HasMostCommonEmojis(FeatureExtractor):

    _n=5
    _n_most_common_emojis=None
    _emojis=[]
    _emoji_re_masks=[]
    def __init__(self, input_column: str,n:int=20):
        super().__init__([input_column], "#{0}_occurs".format(input_column))
        self._n=n
    
    def get_feature_name(self) ->list:
        """
        returns n feature names for the n most common emojis

        Returns
        -------
        names : TYPE
            list.

        """
        suffixes=self._emojis
        names=[]
        for i in range(0,self._n):
            names+=[COLUMN_CONTAINED_EMOJI.format(suffixes[i])]
        return names
    
    def _set_variables(self, inputs: list):
        """
        find out the most common emojis

        Parameters
        ----------
        inputs : list
            List of emojis in tweet.

        Returns
        -------
        None.

        """
        
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
        
        self._emojis=[emoji for (emoji,count) in self._n_most_common_emojis]
        
        #generate mask for each emoji
        for i in self._emojis:
            mask=re.compile(i)
            self._emoji_re_masks+=[mask]
        
        
        print("the most common n emojis are: "+str(self._emojis))

    def _get_values(self, inputs: list) -> np.ndarray :
        """
        returns the amount of occurences of the emoji in the tweet for every emoji.

        Parameters
        ----------
        inputs : list
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        
        
        list_of_most_common_emojis=[]

        #splitting up the input
        tweets=inputs[0]

        
        for tweet in tweets:
            ohe_emojis_used=self._n*[0]
            for i in range(self._n):
                current_emoji=self._emojis[i][0]
                emojis_in_tweet=ast.literal_eval(tweet)
                
                ohe_emojis_used[i]=0
                if current_emoji in emojis_in_tweet:
                    findings=[]
                    findings+=re.findall(self._emoji_re_masks[i],tweet)
                    
                    ohe_emojis_used[i]=len(findings)

            
            #add to list
            list_of_most_common_emojis+=[ohe_emojis_used]
        
        #saving it in an array
        result = np.array(list_of_most_common_emojis)
        return result

    