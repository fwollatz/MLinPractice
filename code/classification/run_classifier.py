#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve


# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-f", "--frequency", action = "store_true", help = "label frequency classifier")
parser.add_argument("-a", "--accuracy", action = "store_true", help = "evaluate using accuracy")
parser.add_argument("-tka", "--topkaccuracy", action = "store_true", help = "evaluate using top k accuracy")
parser.add_argument("-c", "--confusionmatrix", action = "store_true", help = "print the confusion-matrix")
parser.add_argument("-k", "--kappa", action = "store_true", help = "evaluate using Cohen's kappa")
parser.add_argument("-auc","--auc",action = "store_true",help = "evaluate using Area Under ROC curve")
parser.add_argument("-roc","--roc",action = "store_true",help = "show the corresponding ROC curve")
args = parser.parse_args()

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        classifier = pickle.load(f_in)

else:   # manually set up a classifier
    
    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")
        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
        classifier.fit(data["features"], data["labels"])
    elif args.frequency:
        # label frequency classifier
        print("    label frequency classifier")
        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)
        classifier.fit(data["features"], data["labels"])

# now classify the given data
prediction = classifier.predict(data["features"])

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))
if args.topkaccuracy:
    evaluation_metrics.append(("top k accuracy", top_k_accuracy_score))
if args.kappa:
    evaluation_metrics.append(("Cohen's kappa", cohen_kappa_score))
if args.confusionmatrix:
   print("Confusion-Matrix:")
   print(confusion_matrix(data["labels"], prediction))

if args.auc:
    evaluation_metrics.append(("AUC", roc_auc_score))
if args.roc:
    evaluation_metrics.append(("ROC Curve", roc_curve))

    

# compute and print them
for metric_name, metric in evaluation_metrics:
    if metric_name == "ROC Curve" and args.roc:
        fpr,tpr,threshold = metric(data["labels"],prediction)
        plt.title('Receiver Operating Characterics Curve')
        plt.plot(fpr, tpr)
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.show()
    else:
        print("    {0}: {1}".format(metric_name, metric(data["labels"], prediction)))
        
    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(classifier, f_out)