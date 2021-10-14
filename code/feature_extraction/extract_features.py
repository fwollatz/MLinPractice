#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.

Created on Wed Sep 29 11:00:24 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
import numpy as np
from code.feature_extraction.character_length import CharacterLength
from code.feature_extraction.number_of_hashtags import NumberOfHashtags
from code.feature_extraction.number_of_urls import NumberOfURLs
from code.feature_extraction.datetime import DateTime
from code.feature_extraction.hour import Hour  
from code.feature_extraction.month import Month 
from code.feature_extraction.weekday import Weekday
from code.feature_extraction.has_most_common_hashtags import HasMostCommonHashtags
from code.feature_extraction.has_most_common_emojis import HasMostCommonEmojis


from code.feature_extraction.feature_collector import FeatureCollector


from code.util import COLUMN_LABEL, COLUMN_TWEET ,COLUMN_LIKES ,COLUMN_RETWEETS ,COLUMN_TIME,COLUMN_DATE ,COLUMN_HASHTAG ,COLUMN_URLS,COLUMN_PHOTOS ,COLUMN_VIDEOS ,COLUMN_LANGUAGE,COLUMN_EMOJIS


# setting up CLI
parser = argparse.ArgumentParser(description = "Feature Extraction")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
parser.add_argument("-c", "--char_length", action = "store_true", help = "compute the number of characters in the tweet")
parser.add_argument("-ht", "--number_of_hashtags", action = "store_true", help = "compute the number of hashtags")
parser.add_argument("-url", "--number_of_urls", action = "store_true", help = "compute the number of urls")
parser.add_argument("-dt", "--datetime", action = "store_true", help = "compute unix-time of the post")
parser.add_argument("-hr", "--hour", action = "store_true", help = "compute the hour of the day")
parser.add_argument("-mo", "--month", action = "store_true", help = "compute the month of the year")
parser.add_argument("-wd", "--weekday", action = "store_true", help = "compute the day of the week")
parser.add_argument("-f", "--follower", action = "store_true", help = "compute the amount of followers")
parser.add_argument("-ch", "--has_most_common_hashtags", action = "store_true", help = "check wether the tweet has the top n hashtags")
parser.add_argument("-ce", "--has_most_common_emojis", action = "store_true", help = "check wether the tweet has the top n emojis")
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")



if args.import_file is not None:
    # simply import an exisiting FeatureCollector
    with open(args.import_file, "rb") as f_in:
        feature_collector = pickle.load(f_in)

else:    # need to create FeatureCollector manually

    # collect all feature extractors
    features = []
    if args.has_most_common_hashtags :
        #create ohe feature, if the top n most commonly used hashtags are existent
        features.append(HasMostCommonHashtags(COLUMN_HASHTAG,20))
    if args.has_most_common_emojis :
        #create ohe feature, if the top n most commonly used Emojis are existent
        features.append(HasMostCommonEmojis(COLUMN_EMOJIS,20))
    if args.char_length:
        # character length of original tweet (without any changes)
        features.append(CharacterLength(COLUMN_TWEET))
    if args.number_of_urls:
        # amount of urls found in the tweet
        features.append(NumberOfURLs(COLUMN_URLS))
    if args.number_of_hashtags:
        # amount of hashtags found in the tweet
        features.append(NumberOfHashtags(COLUMN_HASHTAG))
    if args.datetime:
        # amount of hashtags found in the tweet
        features.append(DateTime(COLUMN_DATE, COLUMN_TIME))
    if args.month:
        # amount of hashtags found in the tweet
        features.append(Month(COLUMN_DATE))
    if args.weekday:
        # amount of hashtags found in the tweet
        features.append(Weekday(COLUMN_DATE))
    if args.hour:
        # amount of hashtags found in the tweet
        features.append(Hour(COLUMN_TIME))
    if args.hour:
        # amount of hashtags found in the tweet
        features.append(Hour(COLUMN_TIME))
    
    # create overall FeatureCollector
    feature_collector = FeatureCollector(features)
    
    # fit it on the given data set (assumed to be training data)
    feature_collector.fit(df)


# apply the given FeatureCollector on the current data set
# maps the pandas DataFrame to an numpy array
feature_array = feature_collector.transform(df)


# get label array
label_array = np.array(df[COLUMN_LABEL])
label_array = label_array.reshape(-1, 1)

# store the results
results = {"features": feature_array, "labels": label_array, 
           "feature_names": feature_collector.get_feature_names()}

#print(feature_array)
print("we now have "+int(len(results["feature_names"]))+" features!")
print(results["feature_names"])

with open(args.output_file, 'wb') as f_out:
    pickle.dump(results, f_out)

# export the FeatureCollector as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(feature_collector, f_out)