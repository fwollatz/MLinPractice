#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:57:29 2021

@author: ml
"""



from code.feature_extraction.weekday import Weekday
import numpy as np
import pandas as pd
import unittest

class WeekdayTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.Weekday_feature = Weekday(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ['1998-08-21']
    
    def test_input_columns(self):
        """
        tests if the input column is the correct one

        Returns
        -------
        None.

        """
        self.assertEqual(self.Weekday_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        """
        checks if all the feature names have been generated correctly

        Returns
        -------
        None.

        """
        self.assertEqual(self.Weekday_feature.get_feature_name(), ['WEEKDAY_MO', 'WEEKDAY_DI', 'WEEKDAY_MI', 'WEEKDAY_DO', 'WEEKDAY_FR', 'WEEKDAY_SA', 'WEEKDAY_SO'])


    def test_ohe_weekday_correct(self):
        """
        checks if the output is correct

        Returns
        -------
        None.

        """
        output=self.Weekday_feature.fit_transform(self.df)
        
        #CREATE THE ONE HOT ENCODING (OHE)
        #                       0     1     2     3     4    5     6
        #                       MO    DI    MI    DO    FR   SA    SO
        ohe_weekday=np.array([False,False,False,False,True,False,False])


        for i in range(0,len(ohe_weekday)):
            self.assertEqual(output[0][i], ohe_weekday[i])

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_weekday]__")
    unittest.main()