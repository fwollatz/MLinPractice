#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:35:42 2021

@author: ml
"""

import unittest
import pandas as pd
from code.feature_extraction.number_of_words import NumberOfWords
from code.util import COLUMN_WORD_COUNT

class NumberOfWordsTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.number_of_words_extractor = NumberOfWords(self.INPUT_COLUMN)
        
    def test_amount_of_words_correct(self):
        #arrange
        input_texts = ["['First','Second','Third','Fourth']", 
                       "[]"]
        expected_outputs = [4,0]
        
        df = pd.DataFrame()
        
        
        for input_text, expected_output in zip(input_texts, expected_outputs):
            df[self.INPUT_COLUMN] = [input_text]
            #act
            output=self.number_of_words_extractor.fit_transform(df) 
           
            #assert           
            self.assertEqual(output[0][0], expected_output)

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_number_of_words]__")
    unittest.main()