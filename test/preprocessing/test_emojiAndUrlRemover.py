#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 22:36:22 2021

@author: ml
"""

import unittest
import pandas as pd
from code.preprocessing.emoji_url_remover import EmojiAndUrlRemover

class EmojiAndUrlRemoverTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.EMOJI_AND_URL_REMOVER = EmojiAndUrlRemover(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
        
    def test_removing_emoji_works(self):
        #arrange
        input_texts = ["A grinning face \U0001f600", "Squinting face \U0001F606", "Laughing face \U0001F923","No smiley"]
        expected_output_texts = ["A grinning face ", "Squinting face ", "Laughing face ", "No smiley"]
        
        input_df = pd.DataFrame()
        for input_text, expected_text in zip(input_texts,expected_output_texts):
            input_df[self.INPUT_COLUMN] = [input_text]
            
            #act
            emoji_free_text = self.EMOJI_AND_URL_REMOVER.fit_transform(input_df)
            
            #assert
            self.assertEqual(emoji_free_text[self.OUTPUT_COLUMN][0], expected_text)
            
    def test_removing_url_works(self):
        #arrange
        input_texts = ["Link 1: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b", 
                       "Link 2: https://trello.com/b/QXNueqhf/humblebees-777", 
                       "Link 3: http://www.youtube.com/feed/subscriptions",
                       "No Link"]
        expected_output_texts = ["Link 1: ", "Link 2: ", "Link 3: ", "No Link"]
        
        input_df = pd.DataFrame()
        for input_text, expected_text in zip(input_texts,expected_output_texts):
            input_df[self.INPUT_COLUMN] = [input_text]
            
            #act
            emoji_free_text = self.EMOJI_AND_URL_REMOVER.fit_transform(input_df)
            
            #assert
            self.assertEqual(emoji_free_text[self.OUTPUT_COLUMN][0], expected_text)
        
if __name__ == '__main__':
    print("__[RUNNING: test.preprocessing.EmojiAndUrlRemoverTest]__")
    unittest.main()