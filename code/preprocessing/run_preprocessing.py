#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from code.preprocessing.lower_caser import LowerCaser
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.stemmer import Stemmer
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.stop_word_remover import StopWordRemover

from code.preprocessing.emoji_url_remover import EmojiAndUrlRemover
from code.util import COLUMN_TWEET, COLUMN_LANGUAGE
from code.util import COLUMN_LOWERED, COLUMN_STEMMED, COLUMN_TOKENIZED, COLUMN_PUNCTUATION
from code.util import ENGLISCH_TAG, COLUMN_EMOJI_URL, COLUMN_STOP_WORD_REMOVED
import pandas as pd
from sklearn.pipeline import make_pipeline

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-p", "--punctuation", action = "store_false", help = "remove punctuation")
parser.add_argument("-t", "--tokenize", action = "store_false", help = "tokenize given column into individual words")
parser.add_argument("--tokenize_input", help = "input column to tokenize", default = COLUMN_TWEET)
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)

parser.add_argument("-l", "--filter_english", action = "store_false", help = "use only english tagged tweets")
parser.add_argument("-s","--stem", action="store_false", help= "stem the tweets using englisch stemmer")
parser.add_argument("-lc", "--lower_case", action = "store_false", help = "lower cases all tweets")
parser.add_argument("-swr", "--stop_word_removal", action = "store_false", help = "removes all english stop words from the tweets")
parser.add_argument("-feu", "--filter_emojis_urls", action = "store_false", help = "removes emojis and urls from the tweets")
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# filter out non-englisch tagged tweets
if args.filter_english:
    count_before_filtering = len(df.index)
    df = df.loc[df[COLUMN_LANGUAGE] == ENGLISCH_TAG]
    print("{0} tweets were removed when filtering out not-englisch tweets".format((count_before_filtering - len(df.index))))

# collect all preprocessors
preprocessors = []
if args.lower_case:
    preprocessors.append(LowerCaser(COLUMN_TWEET, COLUMN_LOWERED))
if args.punctuation:
    preprocessors.append(PunctuationRemover(COLUMN_LOWERED, COLUMN_PUNCTUATION))
if args.filter_emojis_urls:
    preprocessors.append(EmojiAndUrlRemover(COLUMN_PUNCTUATION, COLUMN_EMOJI_URL))
if args.tokenize:
    preprocessors.append(Tokenizer(COLUMN_EMOJI_URL, COLUMN_TOKENIZED))
if args.stem:
    preprocessors.append(Stemmer(COLUMN_TOKENIZED, COLUMN_STEMMED))
if args.stop_word_removal:
    preprocessors.append(StopWordRemover(COLUMN_STEMMED, COLUMN_STOP_WORD_REMOVED))



# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)

# store the results
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

# create a pipeline if necessary and store it as pickle file (i.e. byte stream)
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
