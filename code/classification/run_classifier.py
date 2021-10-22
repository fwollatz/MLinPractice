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

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from mlflow import log_metric, log_param, set_tracking_uri
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV # to find proper C and gamma of SVM

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
"""-------------- Classifier Choices -------------"""
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-f", "--frequency", action = "store_true", help = "label frequency classifier")
parser.add_argument("--knn", type = int, help = "k nearest neighbor classifier with the specified value of k", default = None)
parser.add_argument("--svc", action = "store_true", help="Support vector classifier")
parser.add_argument("--dtc", action = "store_true", help = "use the decision tree classifier")
parser.add_argument("--dtc_max_depth", type = int, help="decicion tree classifier with the specfied value for the max_deoth", default = None)
parser.add_argument("--dtc_criterion_entropy", action = "store_true", help = "use the entropy crition parameter for the decision tree classifier. Default criterion is 'gini'")
parser.add_argument("--dtc_splitter_random", action = "store_true", help = "use the random splitter parameter for the decision tree classifier. Default splitter is 'best'")
"""-------------- Evaluation Matrix Choices -------------"""
parser.add_argument("-a", "--accuracy", action = "store_true", help = "evaluate using accuracy")
parser.add_argument("-tka", "--topkaccuracy", action = "store_true", help = "evaluate using top k accuracy")
parser.add_argument("-c", "--confusionmatrix", action = "store_true", help = "print the confusion-matrix")
parser.add_argument("-k", "--kappa", action = "store_true", help = "evaluate using Cohen's kappa")
parser.add_argument("-auc","--auc",action = "store_true",help = "evaluate using Area Under ROC curve")
parser.add_argument("-roc","--roc",action = "store_true",help = "show the corresponding ROC curve")
parser.add_argument("--log_folder", help = "where to log the mlflow results", default = "data/classification/mlflow")
args = parser.parse_args()

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

set_tracking_uri(args.log_folder)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        input_dict = pickle.load(f_in)
    
    classifier = input_dict["classifier"]
    for param, value in input_dict["params"].items():
        log_param(param, value)
    
    log_param("dataset", "validation")

else:   # manually set up a classifier
    
    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")
        log_param("classifier", "majority")
        params = {"classifier": "majority"}
        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
        
    elif args.frequency:
        # label frequency classifier
        print("    label frequency classifier")
        log_param("classifier", "frequency")
        params = {"classifier": "frequency"}
        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)
        
    
    elif args.knn is not None:
        print("    {0} nearest neighbor classifier".format(args.knn))
        log_param("classifier", "knn")
        log_param("k", args.knn)
        params = {"classifier": "knn", "k": args.knn}
        standardizer = StandardScaler()
        knn_classifier = KNeighborsClassifier(args.knn, n_jobs = -1)
        classifier = make_pipeline(standardizer, knn_classifier)

    elif args.svc:
        print("   Support vector classifier")
        log_param("classifier", "svc")
        # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
        # https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(data["features"], data["labels"].ravel())
        best_params = grid.best_params_
        kernel = best_params["kernel"]
        C = best_params["C"]
        gamma = best_params["gamma"]
        log_param("kernel", kernel)
        log_param("C", C)
        log_param("gamma", gamma)
        params = {"classifier" : "svc", "kernel": kernel, "C": C, "gamma": gamma}
        classifier = SVC(kernel=kernel, C=C, gamma=gamma)

    elif args.dtc:
        # set default configuration
        criterion_param = "gini"
        splitter_param = "best"
        max_depth_param = None
        # determine console args
        if args.dtc_criterion_entropy:
            criterion_param = "entropy"
        if args.dtc_splitter_random:
            splitter_param = "random"
        if args.dtc_max_depth is not None:
            max_depth_param = args.dtc_max_depth
        print("   Decision Tree Classifier, criterion = {0}, splitter = {1}, max_depth = {2}".format(criterion_param, splitter_param, max_depth_param))
        log_param("classifier", "decision_tree")
        log_param("dt_criterion", criterion_param)
        log_param("dt_spliiter", splitter_param)
        log_param("dt_max_depth", max_depth_param)
        params = {"classifier" : "decision_tree", "dt_criterion" : criterion_param, 
                  "dt_spliiter" : splitter_param, "dt_max_depth" : max_depth_param}
        classifier = DecisionTreeClassifier(criterion = criterion_param,
                                            splitter = splitter_param,
                                            max_depth = max_depth_param)

    classifier.fit(data["features"], data["labels"].ravel())
    log_param("dataset", "training")

# now classify the given data
prediction = classifier.predict(data["features"])

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))
if args.topkaccuracy:
    evaluation_metrics.append(("top k accuracy", top_k_accuracy_score))
if args.kappa:
    evaluation_metrics.append(("Cohen_kappa", cohen_kappa_score))
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
    	metric_value = metric(data["labels"], prediction)
    	print("    {0}: {1}".format(metric_name, metric(data["labels"], prediction)))
    	log_metric(metric_name, metric_value)
    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    output_dict = {"classifier": classifier, "params": params}
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(output_dict, f_out)
