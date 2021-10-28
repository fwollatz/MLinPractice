#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:57:29 2021

@author: ml
"""

from code.feature_extraction.month import Month
import numpy as np
import pandas as pd
import unittest

class MonthTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.Month_feature = Month(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ['1998-08-21']
    
    def test_input_columns(self):
        """
        tests if the input column is the correct one

        Returns
        -------
        None.

        """
        self.assertEqual(self.Month_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        """
        test if feature column is correctly named

        Returns
        -------
        None.

        """
        self.assertEqual(self.Month_feature.get_feature_name(), ['MONTH_JA', 'MONTH_FEB', 'MONTH_MA', 'MONTH_APR', 'MONTH_MAI', 'MONTH_JUN', 'MONTH_JUL', 'MONTH_AUG', 'MONTH_SEP', 'MONTH_OKT', 'MONTH_NOV', 'MONTH_DEZ'])


    def test_ohe_month_correct(self):
        """
        test if the correct month is picked

        Returns
        -------
        None.

        """
        #arrange
        #CREATE THE ONE HOT ENCODING (OHE)
        #                     0      1     2     3    4      5     6     7     8     9     10    11
        #                     JA    FEB   MA    APR   MAI   JUN   JUL   AUG   SEP   OKT   NOV   DEZ
        ohe_august=np.array([False,False,False,False,False,False,False,True,False,False,False,False])
        
        #act
        output=self.Month_feature.fit_transform(self.df)
        
        for i in range(0,len(ohe_august)):
            #assert
            self.assertEqual(output[0][i], ohe_august[i])

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_month]__")
    unittest.main()