#!/bin/bash

# create directory if not yet existing
mkdir -p data/preprocessing/split/

# install all NLTK models
python -m nltk.downloader all

# add labels
# -m: argument is a module name, no file extension should be given: <pkg>.__main__ will be executed by interpreter as main module
echo "  creating labels"
python -m code.preprocessing.create_labels data/raw/ data/preprocessing/labeled.csv

# other preprocessing (removing punctuation etc.)
echo "  general preprocessing"
python -m code.preprocessing.run_preprocessing data/preprocessing/labeled.csv data/preprocessing/preprocessed.csv  -e data/preprocessing/pipeline.pickle -photo --tokenize -stem --punctuation --filter_englisch --lower_case --stop_word_removal --filter_emojis_urls --extract_emojis

# split the data set
echo "  splitting the data set"
python -m code.preprocessing.split_data data/preprocessing/preprocessed.csv data/preprocessing/split/ -s 42
