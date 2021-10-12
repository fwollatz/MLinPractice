#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 16:59:40 2021

@author: ml
"""

import unittest
import pandas as pd
from code.preprocessing.stemmer import Stemmer

class StemmerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.STEMMER = Stemmer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
        
    def test_stemming_works(self):
        #arrange
        input_texts = ["ran",
                       "running",
                       "fisher",
                       "nothing to stem"]
        expected_output_texts = ["['ran']", "['run']", "['fisher']", "['nothingtostem']"]
        
        input_df = pd.DataFrame()
        for input_text, expected_text in zip(input_texts,expected_output_texts):
            input_df[self.INPUT_COLUMN] = [input_text]
            
            #act
            stemmed_text = self.STEMMER.fit_transform(input_df)
            
            #assert
            self.assertEqual(stemmed_text[self.OUTPUT_COLUMN][0], expected_text)
    
if __name__ == '__main__':
    print("__[RUNNING: test.preprocessing.StemmerTest]__")
    unittest.main()