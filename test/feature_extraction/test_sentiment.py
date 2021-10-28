#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 20:36:30 2021

@author: ml
"""


from code.feature_extraction.sentiment_feature import SentimentFeature
import numpy as np
import pandas as pd
import unittest


class TestSentiment(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.sentiment_extractor = SentimentFeature(self.INPUT_COLUMN)
        
    def test_sentiment_correct(self):
        """
        tests if the sentiments are correctly estimated

        Returns
        -------
        None.

        """
        #arrange
        input_tweets = ["I like cheese very very much",
                        "I like cheese",
                        "I dont like cheese",
                        "I hate cheese",
                        "I hate cheese very very much"]

        expected_outputs = [[round(1.3612,4)],
                            [round(1.3612,4)],
                            [round(0.7245,4)],
                            [round(0.4281,4)],
                            [round(0.4281,4)]
                            ]
        
        expected_outputs = np.array(expected_outputs)
        
        df = pd.DataFrame()
        
        df[self.INPUT_COLUMN] = input_tweets
        
        #act
        output = self.sentiment_extractor.fit_transform(df)
        
        #assert     
        for i in range(0,len(output)):
            self.assertEqual(round(output[i][0],4), expected_outputs[i])

if __name__ == '__main__':
    print("__[RUNNING: test.feature_extraction.test_sentiment]__")
    unittest.main()