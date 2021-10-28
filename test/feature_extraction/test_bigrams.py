#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""


from code.feature_extraction.bigrams import BigramFeature
import nltk
import pandas as pd
import unittest

class BigramFeatureTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.bigram_feature = BigramFeature(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ['["This", "is", "a", "tweet", "This", "is", "also", "a", "test"]']
    
    def test_input_columns(self):
        """
        tests if the input column is the correct one

        Returns
        -------
        None.

        """
        self.assertEqual(self.bigram_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        """
        test if feature column is correctly named

        Returns
        -------
        None.

        """
        self.assertEqual(self.bigram_feature.get_feature_name(), self.INPUT_COLUMN + "_bigrams")

    def test_list_of_bigrams_exists(self):
        """
        Check if the list of bigrams exists

        Returns
        -------
        None.

        """
        #act
        self.bigram_feature.fit(self.df)
        #assert
        self.assertGreaterEqual(len(list(self.bigram_feature._bigrams)), 0)

    def test_list_of_bigrams_most_frequent_correct(self):
        """
        checks if the list of most frequent bigrams is correct

        Returns
        -------
        None.

        """
        #arrange
        self.bigram_feature.fit(self.df)
        EXPECTED_BIGRAM = ('This', 'is')
        
        #act
        freq_dist = nltk.FreqDist(self.bigram_feature._bigrams)
        freq_list = list(freq_dist.items())
        freq_list.sort(key = lambda x: x[1], reverse = True)
        
        #assert
        self.assertEqual(freq_list[0][0], EXPECTED_BIGRAM)

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.BigramFeatureTest]__")
    unittest.main()