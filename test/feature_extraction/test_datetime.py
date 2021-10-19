#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:57:29 2021

@author: ml
"""

import unittest
import pandas as pd
from code.feature_extraction.datetime import DateTime
from code.util import COLUMN_DATETIME_UNIX

class DatetimeTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN_DATE = "date"
        self.INPUT_COLUMN_TIME = "time"
        self.datetime_feature = DateTime(self.INPUT_COLUMN_DATE, self.INPUT_COLUMN_TIME)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN_DATE] = ['1998-08-21']
        self.df[self.INPUT_COLUMN_TIME] = ['01:37:08']
    
    def test_input_date_columns(self):
        self.assertEqual(self.datetime_feature._input_columns[0], self.INPUT_COLUMN_DATE)
    
    def test_input_time_columns(self):
        self.assertEqual(self.datetime_feature._input_columns[1], self.INPUT_COLUMN_TIME)

    def test_feature_name(self):
        self.assertEqual(self.datetime_feature.get_feature_name(), COLUMN_DATETIME_UNIX)


    def test_ohe_datetime_correct(self):
        output=self.datetime_feature.fit_transform(self.df)
        
        expected=903656228
        self.assertEqual(output[0][0], expected)

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_datetime]__")
    unittest.main()