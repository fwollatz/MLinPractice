#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 17:23:36 2021

@author: ml
"""
from code.preprocessing.stop_word_remover import StopWordRemover
import pandas as pd
import unittest


class StopWordRemoverTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.STOP_WORD_REMOVER = StopWordRemover(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
        
    def test_removing_stop_words_works(self):
        """
        check if the removal of the stop words worked.

        Returns
        -------
        None.

        """
        #arrange
        input_texts = ["['this','has','some','stop','words']",
                       "['zero','stop','words','removed']"
                       ]
        expected_output_texts = ["['stop', 'words']", "['zero', 'stop', 'words', 'removed']"]
        
        input_df = pd.DataFrame()
        for input_text, expected_text in zip(input_texts,expected_output_texts):
            input_df[self.INPUT_COLUMN] = [input_text]
            
            #act
            no_stop_word_text = self.STOP_WORD_REMOVER.fit_transform(input_df)
            
            #assert
            self.assertEqual(no_stop_word_text[self.OUTPUT_COLUMN][0], expected_text)
    
if __name__ == '__main__':
    print("__[RUNNING: test.preprocessing.StopWordRemoverTest]__")
    unittest.main()