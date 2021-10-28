#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emoji and URL Preprocessor 

masks are from:
   emoji: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
       + https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
   url: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python

Created on Fri Oct  8 12:08:34 2021

@author: ml
"""

from code.preprocessing.preprocessor import Preprocessor
import re

class EmojiExtractor(Preprocessor):
    """collects emojis that have been used in a tweet in a seperate column"""
    
    def __init__(self, input_column, output_column):
        """Initialize the EXTRACTOR with the given input and output column."""
        super().__init__([input_column], output_column)
    
    
    def _set_variables(self, inputs : list):
        """
        setting filtering regex mask for emojis 
        

        Parameters
        ----------
        inputs : list
            list of all tweets.

        Returns
        -------
        None.
        """
        
        #store emoji regex mask
        self._emoji_re_mask = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)

    
    def _get_values(self, inputs : list) -> list:
        """
        extracting emojis from tweets

        Parameters
        ----------
        inputs : list
            list of all tweets.

        Returns
        -------
        list
            list of emojis per tweet.

        """
        emojis_filtered = []

        for tweet in inputs[0]:
            #find all emojis
            findings=re.findall(self._emoji_re_mask,tweet)
            
            emojis_filtered.append(str(findings))
        return emojis_filtered