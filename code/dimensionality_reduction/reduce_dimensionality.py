#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a dimensionality reduction technique.

Created on Wed Sep 29 13:33:37 2021

@author: lbechberger
"""

import argparse
from code.dimensionality_reduction.pca_reducer import PCAReducer
from code.dimensionality_reduction.select_k_best_reducer import SelectKBestReducer
from code.dimensionality_reduction.rfe_reducer import RFEReducer
import pickle


# setting up CLI
parser = argparse.ArgumentParser(description = "Dimensionality reduction")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
parser.add_argument("-m", "--mutual_information", type = int, help = "select K best features with Mutual Information", default = None)
parser.add_argument("--pca", action = "store_true", help = "perform Princinple Component Analysis with automated component selection (95% explained variance)")
parser.add_argument("--verbose", action = "store_true", help = "print information about feature selection process")
parser.add_argument("--rfe", type = int, help = "perform feature selection (select n best) by using a wrapper method. Use an additional argument to specify the model (default is Decision Tree Classifier). This option is only available for Decision Tree and Random Forest Classifier", default = None)
parser.add_argument("-rfe_rfc","--rfe_random_forest_classifier", action = "store_true", help = "use random forest classifier for RFE")
args = parser.parse_args()

# load the data
with open(args.input_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

features = input_data["features"]
labels = input_data["labels"]
feature_names = input_data["feature_names"]

if args.import_file is not None:
    # simply import an already fitted dimensionality reducer
    with open(args.import_file, 'rb') as f_in:
        dim_red = pickle.load(f_in)

else: # need to set things up manually

    if args.mutual_information is not None:
        # select K best based on Mutual Information
        dim_red = SelectKBestReducer(features, labels)
        dim_red.fit(k = args.mutual_information)
        
        if args.verbose:
            print("    SelectKBest with Mutual Information and k = {0}".format(args.mutual_information))
            print("    {0}".format(feature_names))
            print("    " + str(dim_red.get_scores()))
            print("    " + str(dim_red.get_feature_names(feature_names)))
    elif args.pca:
        #use pca for dim reduction
        dim_red = PCAReducer(features, labels)
        dim_red.fit()
    elif args.rfe is not None:
        model = "dtc"
        if args.rfe_random_forest_classifier:
            model = "rfc"
        n_best_features = args.rfe
        print("    RFE with model = {0} and n_best_features = {1}".format(model, n_best_features))
        dim_red = RFEReducer(features, labels, feature_names, model, n_best_features)
        dim_red.fit()
    pass

# apply the dimensionality reduction to the given features
reduced_features = dim_red.transform(features)

# store the results
output_data = {"features": reduced_features, 
               "labels": labels}
with open(args.output_file, 'wb') as f_out:
    pickle.dump(output_data, f_out)
      

# export the dimensionality reduction technique as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(dim_red, f_out)