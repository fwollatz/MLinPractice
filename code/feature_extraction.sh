#!/bin/bash

# create directory if not yet existing
mkdir -p data/feature_extraction/

# run feature extraction on training set (may need to fit extractors)
echo "  training set"
python -m code.feature_extraction.extract_features data/preprocessing/split/training.csv data/feature_extraction/training.pickle -e data/feature_extraction/pipeline.pickle --char_length --number_of_hashtags --number_of_urls --datetime --hour --month --weekday --has_most_common_hashtags --has_most_common_emojis --photo --follower

# run feature extraction on validation set and test set (with pre-fit extractors)
echo "  validation set"
python -m code.feature_extraction.extract_features data/preprocessing/split/validation.csv data/feature_extraction/validation.pickle -i data/feature_extraction/pipeline.pickle
echo "  test set"
python -m code.feature_extraction.extract_features data/preprocessing/split/test.csv data/feature_extraction/test.pickle -i data/feature_extraction/pipeline.pickle
