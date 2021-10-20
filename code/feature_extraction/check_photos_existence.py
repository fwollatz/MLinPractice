#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that checks if photo(s) exists or not
@return: 0 for no photos, 1 for photo(s) included

Created on Thur Oct 07 2021

@author: RCheng
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_PHOTO_EXISTENCE

class PhotoChecker(FeatureExtractor):
    # constructor
    def __init__(self, input_column: str):
        """
        constructor
        :param input_column: name of the input column) - photos

        """
        super().__init__([input_column], COLUMN_PHOTO_EXISTENCE)

    # set internal variables based on input columns
    def _set_variables(self, inputs):
        pass

    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs) ->list:
        """
        calculate if a tweet contains any photos
        :param inputs:
        :return: a list of bool, whether a tweet contains any photos
        """
        result = inputs[0].str.contains('https://pbs.twimg.com/|.png|.jpg').astype(int).values
        result = result.reshape(-1, 1)
        return result