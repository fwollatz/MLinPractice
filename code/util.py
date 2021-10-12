#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""

# suffixes for general preprocessing
SUFFIX_REMOVED_PUNCTUATION = "_no_punctuation"
SUFFIX_EMOJI_URL = "_no_emojis_urls"
SUFFIX_TOKENIZED = "_tokenized"
SUFFIX_STEMMED = "_stemmed"
SUFFIX_LOWERCASED = "_lowercased"
SUFFIX_STOP_WORD_REMOVED = "_no_stop_words"



# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"
COLUMN_TIME = "time"
COLUMN_DATE = "date"
COLUMN_HASHTAG = "hashtags"
COLUMN_URLS = "urls"
COLUMN_PHOTOS = "photos"
COLUMN_VIDEOS = "video"
COLUMN_LANGUAGE = "language"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"
SUFFIX_TOKENIZED = "_tokenized"
COLUMN_PHOTO_EXISTENCE = "contain_photos"


# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
# current order of general preprocessing: lower casing > punctuation > tokenization > stemming > stop word removal
COLUMN_LOWERED = COLUMN_TWEET + SUFFIX_LOWERCASED
COLUMN_PUNCTUATION = COLUMN_LOWERED + SUFFIX_REMOVED_PUNCTUATION
COLUMN_EMOJI_URL = COLUMN_PUNCTUATION + SUFFIX_EMOJI_URL
COLUMN_TOKENIZED = COLUMN_EMOJI_URL + SUFFIX_TOKENIZED
COLUMN_STEMMED = COLUMN_TOKENIZED + SUFFIX_STEMMED
COLUMN_STOP_WORD_REMOVED = COLUMN_STEMMED + SUFFIX_STOP_WORD_REMOVED
# column with all general preprocessing done
COLUMN_GENERAL_PREPROCESSED = COLUMN_STOP_WORD_REMOVED



ENGLISCH_TAG = "en"

#Column names of Features
COLUMN_URL_COUNT="URL_Count"
COLUMN_HASHTAG_COUNT="Hashtag_Count"
COLUMN_DATETIME_UNIX="datetime_unix"
COLUMN_HOUR="HOUR OF DAY"
COLUMN_WEEKDAY="WEEKDAY"
COLUMN_MONTH="MONTH"

#common methods
def string_to_words_list(string_list : str) -> list:
    """
    Converts a string list: "['word1', 'word2', 'word3']" 
    into a list: ['word1','word2','word3']
    """
    helper_str = str(string_list).replace("[","").replace("]","").replace("'","").replace(" ","")
    words = helper_str.split(",")
    return words
