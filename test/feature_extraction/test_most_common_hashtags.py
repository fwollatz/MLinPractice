#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 20:36:30 2021

@author: ml
"""

import unittest
import pandas as pd
import numpy as np
from code.feature_extraction.has_most_common_hashtags import HasMostCommonHashtags

class HasMostCommonHashtagsTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.has_most_common_hashtag_extractor = HasMostCommonHashtags(self.INPUT_COLUMN, n = 2)
        
    def test_top_most_common_hashtags_correct(self):
        #arrange
        input_tweets = ["['Top1','Top1','Top2','Top2']",
                        "['Top1','Top1','Top1','Top3']",
                        "['Top2','Top3']",
                        "['Top4']"]

        expected_outputs = [[True, True],
                            [True, False],
                            [False, True],
                            [False, False]]
        expected_outputs = np.array(expected_outputs)
        
        df = pd.DataFrame()
        
        df[self.INPUT_COLUMN] = input_tweets
        
        #act
        output = self.has_most_common_hashtag_extractor.fit_transform(df)
        
        #assert     
        self.assertTrue((output == expected_outputs).all())
        
if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_has_most_common_hashtags]__")
    unittest.main()