#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:57:29 2021

@author: ml
"""

from code.feature_extraction.hour import Hour
import numpy as np
import pandas as pd
import unittest

class HourTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.Hour_feature = Hour(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ['01:37:00']
    
    def test_input_columns(self):
        """
        tests if the input column is the correct one

        Returns
        -------
        None.

        """
        self.assertEqual(self.Hour_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        """
        test if feature column is correctly named

        Returns
        -------
        None.

        """
        self.assertEqual(self.Hour_feature.get_feature_name(), ['HOUR OF DAY_0-2', 'HOUR OF DAY_3-5', 'HOUR OF DAY_6-8', 'HOUR OF DAY_9-11', 'HOUR OF DAY_12-14', 'HOUR OF DAY_15-17', 'HOUR OF DAY_18-20', 'HOUR OF DAY_21-23'])


    def test_ohe_hour_correct(self):
        """
        check if hour is correctly checked

        Returns
        -------
        None.

        """
        #arrange
        #CREATE THE ONE HOT ENCODING (OHE)
        #                    0      1     2     3    4      5     6     7     
        #                   0-2    3-5  6-8  9-11 12-14 15-17 18-20 21-23
        ohe_hour=np.array([True,False,False,False,False,False,False,False])
        
        #act
        output=self.Hour_feature.fit_transform(self.df)
        
        
        
        for i in range(0,len(ohe_hour)):
            #assert
            self.assertEqual(output[0][i], ohe_hour[i])

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_hour]__")
    unittest.main()