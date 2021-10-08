#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor for stemming

Created on Thu Oct  7 14:00:58 2021

@author: zqirui
"""

from code.preprocessing.preprocessor import Preprocessor
from code.util import string_to_words_list
import nltk

class Stemmer(Preprocessor):
    

    def __init__(self, input_column, output_column):
        """
        Constructor
        """
        # input column "tweet", new output column
        super().__init__([input_column], output_column)
        
    
    def _get_values(self, inputs : list) -> list:
        """
        Stems a tweet
        """
        stemmed = []
        
        stemmer = nltk.stem.snowball.SnowballStemmer("english")
        for tweet in inputs[0]:
            tweet_stemmed = []
            words = string_to_words_list(tweet)
            for word in words:
                word_stemmed = stemmer.stem(word)
                tweet_stemmed.append(word_stemmed)
            stemmed.append(str(tweet_stemmed))
        print("{0} tweets were stemmed!".format(len(stemmed)))
        return stemmed
    
