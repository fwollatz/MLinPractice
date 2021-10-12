#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 23:11:33 2021

@author: ml
"""

import unittest
import pandas as pd
from code.preprocessing.lower_caser import LowerCaser

class LowerCaserTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.LOWER_CASER = LowerCaser(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
        
    def test_lower_casing_works(self):
        #arrange
        input_texts = ["A B C D", 
                       "ThIs Is A tExT wItH mIxEd UpPeR aNd LoWeR cAsE", 
                       "nothing to lower"]
        expected_output_texts = ["a b c d", "this is a text with mixed upper and lower case", "nothing to lower"]
        
        input_df = pd.DataFrame()
        for input_text, expected_text in zip(input_texts,expected_output_texts):
            input_df[self.INPUT_COLUMN] = [input_text]
            
            #act
            lowered_text = self.LOWER_CASER.fit_transform(input_df)
            
            #assert
            self.assertEqual(lowered_text[self.OUTPUT_COLUMN][0], expected_text)
    
if __name__ == '__main__':
    print("__[RUNNING: test.preprocessing.LowerCaserTest]__")
    unittest.main()