#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:30:41 2021

@author: ml
"""


from code.preprocessing.tokenizer import Tokenizer
import pandas as pd
import unittest



class TokenizerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.tokenizer = Tokenizer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    def test_input_columns(self):
        """
        check if the input column is correctly set

        Returns
        -------
        None.

        """
        self.assertListEqual(self.tokenizer._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        """
        check if the output colum is correctly set

        Returns
        -------
        None.

        """
        self.assertEqual(self.tokenizer._output_column, self.OUTPUT_COLUMN)

    def test_tokenization_single_sentence(self):
        """
        check if the tokenization is correct

        Returns
        -------
        None.

        """
        input_text = "This is an example sentence"
        output_text = "['This', 'is', 'an', 'example', 'sentence']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        tokenized = self.tokenizer.fit_transform(input_df)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text)
    

if __name__ == '__main__':
    print("__[RUNNING: test.preprocessing.TokenizerTest]__")
    unittest.main()