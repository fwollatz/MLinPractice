#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 00:56:02 2021

@author: ml
"""


from code.preprocessing.punctuation_remover import PunctuationRemover
import pandas as pd
import unittest


class PunctuationRemoverTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.PUNCTUATION_REMOVER = PunctuationRemover(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
        
    def test_removing_punctuation_works(self):
        """
        test if the removal of the puctution has worked

        Returns
        -------
        None.

        """
        #arrange
        input_texts = ["A,B,C,D", 
                       "!#$%&'()*+,-./:;<=>?@[]^_`{|}~", 
                       "nothing to remove"]
        expected_output_texts = ["ABCD", "", "nothing to remove"]
        
        input_df = pd.DataFrame()
        for input_text, expected_text in zip(input_texts,expected_output_texts):
            input_df[self.INPUT_COLUMN] = [input_text]
            
            #act
            no_punctuation_text= self.PUNCTUATION_REMOVER.fit_transform(input_df)
            
            #assert
            self.assertEqual(no_punctuation_text[self.OUTPUT_COLUMN][0], expected_text)
    
if __name__ == '__main__':
    print("__[RUNNING: test.preprocessing.PunctuationRemoverTest]__")
    unittest.main()