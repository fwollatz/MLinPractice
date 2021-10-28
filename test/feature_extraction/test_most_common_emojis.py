#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 20:36:30 2021

@author: ml
"""


from code.feature_extraction.has_most_common_emojis import HasMostCommonEmojis
import numpy as np
import pandas as pd
import unittest

class HasMostCommonWordsEmojis(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.has_most_common_emojis_extractor = HasMostCommonEmojis(self.INPUT_COLUMN, n = 2)
        
    def test_top_most_common_emojis_correct(self):
        """
        check if the emojis are counted correctly

        Returns
        -------
        None.

        """
        #arrange
        input_tweets = ['["💀","💀"]',
                        '["🎃","🎃"]',
                        '["🎃"]',
                        '["👻"]']

        expected_outputs = [[0, 2],
                            [2, 0],
                            [1, 0],
                            [0, 0]]
        expected_outputs = np.array(expected_outputs)
        
        df = pd.DataFrame()
        
        df[self.INPUT_COLUMN] = input_tweets
        
        #act
        output = self.has_most_common_emojis_extractor.fit_transform(df)
        
        #assert     
        self.assertTrue((output == expected_outputs).all())
        
if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_has_most_common_emojis]__")
    unittest.main()