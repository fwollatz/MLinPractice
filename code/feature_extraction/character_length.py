#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of characters in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class CharacterLength(FeatureExtractor):
    
    def __init__(self, input_column: str):
        """
        constructor

        Parameters
        ----------
        input_column : str
            name of the input column

        Returns
        -------
        None.

        """

        super().__init__([input_column], "{0}_charlength".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    

    def _get_values(self, inputs: list) -> np.ndarray :
        """
        compute the word length based on the inputs
        
        Parameters
        ----------
        inputs : list
            list of all tweets

        Returns
        -------
        result : np.ndarray
            array with the character-length of the individual tweets

        """
        
        result = np.array(inputs[0].str.len())
        result = result.reshape(-1,1)
        return result
