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
SUFFIX_HOURS=["0-2","3-5","6-8","9-11","12-14","15-17","18-20","21-23"]
SUFFIX_WEEKDAY=["MO","DI","MI","DO","FR","SA","SO"]
SUFFIX_MONTHS=["JA","FEB","MA","APR","MAI","JUN","JUL","AUG","SEP","OKT","NOV","DEZ"]

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
COLUMN_USERNAME = "username"
COLUMN_USER_ID = "user_id"

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
COLUMN_EMOJIS="Emojis_in_tweet"

PCA_EXPLAINED_VARIANCE_THRESHOLD = 0.95
ENGLISCH_TAG = "en"

#Column names of Features
COLUMN_URL_COUNT="URL_Count"
COLUMN_HASHTAG_COUNT="Hashtag_Count"
COLUMN_DATETIME_UNIX="datetime_unix"
COLUMN_HOUR="HOUR OF DAY"
COLUMN_WEEKDAY="WEEKDAY"
COLUMN_MONTH="MONTH"
COLUMN_FOLLOWER_COUNT="follower_count"
COLUMN_CONTAINED_HASHTAG="contains_#{0}"
COLUMN_CONTAINED_EMOJI="contains_emoji_{0}"
COLUMN_PHOTO_EXISTENCE = "contain_photos"
COLUMN_WORD_COUNT = "word_count"


#common methods
def string_to_words_list(string_list : str) -> list:
    """
    Converts a string list: "['word1', 'word2', 'word3']" 
    into a list: ['word1','word2','word3']
    """
    helper_str = str(string_list).replace("[","").replace("]","").replace("'","").replace(" ","")
    words = helper_str.split(",")
    return words
