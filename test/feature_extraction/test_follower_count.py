#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:38:07 2021

@author: ml
"""


import unittest
import pandas as pd
from code.feature_extraction.follower_count import FollowerCount
from code.util import COLUMN_FOLLOWER_COUNT

class FollowerCountTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.follower_count_feature = FollowerCount(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = [817369124519153664]
    
    def test_input_columns(self):
        self.assertEqual(self.follower_count_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.follower_count_feature.get_feature_name(), COLUMN_FOLLOWER_COUNT)


    def test_ohe_follower_count_correct(self):
        output=self.follower_count_feature.fit_transform(self.df)

        print("INFORMATION: If this test fails, that may be because the twitteraccount @LucasBechberger got more followers!")
        #The Twitter Useraccout @LucasBechberger, id= 817369124519153664 had 195 followers at 15.10.2021;15:42 
        self.assertEqual(output[0][1], 195)

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_follower_count]__")
    unittest.main()