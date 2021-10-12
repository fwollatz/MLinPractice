#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
from code.feature_extraction.number_of_urls import NumberOfURLs
from code.util import COLUMN_URL_COUNT

class BigramFeatureTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.URL_feature = NumberOfURLs(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ['["www.WeAreAwesome.de", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"]']
    
    def test_input_columns(self):
        self.assertEqual(self.URL_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.URL_feature.get_feature_name(), COLUMN_URL_COUNT)


    def test_amount_of_urls_correct(self):
        output=self.URL_feature.fit_transform(self.df)
        EXPECTED_COUNT = 2
        
        self.assertEqual(output[0][0], EXPECTED_COUNT)

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_number_of_urls]__")
    unittest.main()