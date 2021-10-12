#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:57:29 2021

@author: ml
"""

import unittest
import pandas as pd
from code.feature_extraction.month import Month
from code.util import COLUMN_MONTH
import numpy as np

class BigramFeatureTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.Month_feature = Month(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ['1998-08-21']
    
    def test_input_columns(self):
        self.assertEqual(self.Month_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.Month_feature.get_feature_name(), COLUMN_MONTH)


    def test_ohe_month_correct(self):
        output=self.Month_feature.fit_transform(self.df)
        
        #CREATE THE ONE HOT ENCODING (OHE)
        #                     0      1     2     3    4      5     6     7     8     9     10    11
        #                     JA    FEB   MA    APR   MAI   JUN   JUL   AUG   SEP   OKT   NOV   DEZ
        ohe_august=np.array([False,False,False,False,False,False,False,True,False,False,False,False])


        for i in range(0,len(ohe_august)):
            self.assertEqual(output[0][i], ohe_august[i])

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_month]__")
    unittest.main()