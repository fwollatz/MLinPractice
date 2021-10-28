#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:38:07 2021

@author: ml
"""

from code.feature_extraction.follower_count import FollowerCount
from code.util import COLUMN_FOLLOWER_COUNT
import pandas as pd
import pickle
import unittest

class FollowerCountTest(unittest.TestCase):

    def setUp(self):
        #remove the id from the file to check for current results
        try:
            a_file = open("id_to_follower.pkl", "rb")
            id_to_follower = pickle.load(a_file)
            a_file.close()
        except:
            print("The file could not be loaded. Try to recreate the id_to_follower dict. This IS GOING TO take forever.")
            id_to_follower={}
        try:
            id_to_follower.pop(401048959)
        except KeyError:
            print("Es gab den key nicht")
        
        a_file = open("id_to_follower.pkl", "wb")
        pickle.dump(id_to_follower, a_file)
        a_file.close()
        
        self.INPUT_COLUMN = "input"
        self.follower_count_feature = FollowerCount(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = [401048959]
        
        
    
    def test_input_columns(self):
        """
        tests if the input column is the correct one

        Returns
        -------
        None.

        """
        self.assertEqual(self.follower_count_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        """
        test if feature column is correctly named

        Returns
        -------
        None.

        """
        self.assertEqual(self.follower_count_feature.get_feature_name(), COLUMN_FOLLOWER_COUNT)


    def test_ohe_follower_count_correct(self):
        """
        test if the follower count is correct. Is error prone, since there are discrepancies between servers (see this: https://twittercommunity.com/t/discrepancy-in-followers-count/80018/23)

        Returns
        -------
        None.

        """
        #act
        output=self.follower_count_feature.fit_transform(self.df)

        #assert
        print("INFORMATION: If this test fails, that may be because the twitteraccount @Qubole got more followers or because of discrepancies between servers (see this: https://twittercommunity.com/t/discrepancy-in-followers-count/80018/23)!")
        #The Twitter Useraccout @Qubole, id= 401048959 had 10192 followers at 22.10.2021;21:42 
        self.assertEqual(output[0][0], 10197)

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_follower_count]__")
    unittest.main()