#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

from code.feature_extraction.number_of_hashtags import NumberOfHashtags
from code.util import COLUMN_HASHTAG_COUNT
import pandas as pd
import unittest

class NumberOfHastagsTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.hashtag_feature = NumberOfHashtags(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ['["#WeAreAwesome", "#THISISQUITENICE", "#CamelCasingIsEvil"]']
    
    def test_input_columns(self):
        """
        tests if the input column is the correct one

        Returns
        -------
        None.

        """
        self.assertEqual(self.hashtag_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        """
        test if feature column is correctly named

        Returns
        -------
        None.

        """
        self.assertEqual(self.hashtag_feature.get_feature_name(), COLUMN_HASHTAG_COUNT)



    def test_amount_of_hashtags_correct(self):
        """
        test if the amount of hashtags is correctly counted

        Returns
        -------
        None.

        """
        #act
        output=self.hashtag_feature.fit_transform(self.df)
        EXPECTED_COUNT = 3
        
        #assert
        self.assertEqual(output[0][0], EXPECTED_COUNT)

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_number_of_hashtags]__")
    unittest.main()