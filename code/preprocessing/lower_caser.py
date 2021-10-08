#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor for lower casing

Created on Fri Oct  8 10:01:28 2021

@author: ml
"""

from code.preprocessing.preprocessor import Preprocessor

class LowerCaser(Preprocessor):
    

    def __init__(self, input_column, output_column):
        """
        Constructor
        """
        # input column "tweet", new output column
        super().__init__([input_column], output_column)
        
    
    def _get_values(self, inputs : list) -> list:
        """
        lower cases a string
        """
        column = inputs[0].str.lower()
        print("Lower cased {0} tweets!".format(len(column)))
        return column


