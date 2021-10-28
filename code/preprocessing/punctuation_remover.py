#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes punctuation from the original tweet text.

Created on Wed Sep 29 09:45:56 2021

@author: lbechberger
"""


from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_TWEET, COLUMN_PUNCTUATION
from pandas.core.series import Series
import string

# removes punctuation from the original tweet
# inspired by https://stackoverflow.com/a/45600350
class PunctuationRemover(Preprocessor):
    

    def __init__(self, input_column, output_column):
        """
        Constructor
        """
        # input column "tweet", new output column
        super().__init__([input_column], output_column)
    
    def _set_variables(self, inputs : list):
        """
        set internal variables based on input columns

        Parameters
        ----------
        inputs : list
            list of all tweets.

        Returns
        -------
        None.

        """
        # store punctuation for later reference
        self._punctuation = "[{}]".format(string.punctuation)
    
    def _get_values(self, inputs : list) -> Series:
        """
        get preprocessed column based on data frame and internal variables
        

        Parameters
        ----------
        inputs : list
            list of all tweets.

        Returns
        -------
        Series
            list of all tweets without punctuation.

        """
        # replace punctuation with empty string
        column = inputs[0].str.replace(self._punctuation, "")
        print("Removed punctuation from {0} tweets!".format(len(column)))
        return column