#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:00:36 2021

@author: ml
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_SENTIMENT, string_to_words_list
import numpy as np
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

#class for extracting the amount of words after general preprocessing
class SentimentFeature(FeatureExtractor):
    sentiment_analyzer=SentimentIntensityAnalyzer()
    
    
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

        nltk.download('vader_lexicon')
        super().__init__([input_column], COLUMN_SENTIMENT)
        
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
        
        
        list_of_sentiments=[]
        
        for tweet in inputs[0]:
            sentiment=self.sentiment_analyzer.polarity_scores(tweet)["compound"]+1 #the +1 is to avoid that classifiers have a problem with negative numbers
            list_of_sentiments+=[sentiment]
        #saving it in an array
        result = np.array(list_of_sentiments)
        #expand dim to (tweets, 1)
        result = result.reshape(-1,1)
        print(result)
        return result
    