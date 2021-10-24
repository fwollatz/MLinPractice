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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve
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
parser.add_argument("-mc", "--minority", action = "store_true", help = "minority class classifier")
#KNN
parser.add_argument("--knn", type = int, help = "k nearest neighbor classifier with the specified value of k", default = None)
#Complement Naive Bayes
parser.add_argument("-cnb", "--complement_naive_bayes", action = "store_true", help = "a naive bayes classifier, especially good with imbalanced data. Described in Rennie et al. (2003).")
parser.add_argument("-cnb_a", "--complement_naive_bayes_alpha", type = float , help = "Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).", default=1.0)
parser.add_argument("-cnb_fp", "--complement_naive_bayes_fit_prior", action = "store_false", help = "Only used in edge case with a single class in the training set.")
parser.add_argument("-cnb_n", "--complement_naive_bayes_norm", action = "store_true", help = "Whether or not a second normalization of the weights is performed. The default behavior mirrors the implementations found in Mahout and Weka, which do not follow the full algorithm described in Table 9 of the paper.")
#Support Vector Classifier
parser.add_argument("--svc", action = "store_true", help="Support vector classifier")
parser.add_argument("--svc_c", type = float, help="Specify Regularization parameter C of SVC", default = 1.0)
parser.add_argument("--svc_gamma", type = float, help="Specify Gamma parameter of SVC", default = 1.0)
parser.add_argument("--svc_kernel", help = "Specify the kernel mode of SVC", default = 'rbf')
#Decision Tree Classifier
parser.add_argument("--dtc", action = "store_true", help = "use the decision tree classifier")
parser.add_argument("--dtc_max_depth", type = int, help="decicion tree classifier with the specified value for the max_depth", default = None)
parser.add_argument("--dtc_criterion_entropy", action = "store_true", help = "use the entropy crition parameter for the decision tree classifier. Default criterion is 'gini'")
parser.add_argument("--dtc_splitter_random", action = "store_true", help = "use the random splitter parameter for the decision tree classifier. Default splitter is 'best'")
#Random Forest Classifier
parser.add_argument("--rfc", action = "store_true", help = "use the random forest classifier")
parser.add_argument("--rfc_no_bootstrap", action = "store_true", help = "disable bootstrapping, use the whole dataset for each tree")
parser.add_argument("--rfc_criterion_entropy", action = "store_true", help = "use the entropy crition parameter for the random forest classifier. Default criterion is 'gini'" )
parser.add_argument("--rfc_max_depth", type = int, help = "random forest classifier with the specified value for the max_depth", default = None)
parser.add_argument("--rfc_n_estimators", type = int, help = "random forest classifier with the specified value for the number of trees in the forest. Default is 100", default = 100)
#Balanced Weights Options, available for DTC and RFC 
parser.add_argument("--class_weight_balanced", action = "store_true", help = "use the class weight = 'balanced' option if available to even out inequal label distributions")
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

    # majority vote classifier
    if args.majority:
        print("    majority vote classifier")
        log_param("classifier", "majority")
        params = {"classifier": "majority"}
        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
        
    # label frequency classifier
    elif args.frequency:
        print("    label frequency classifier")
        log_param("classifier", "frequency")
        params = {"classifier": "frequency"}
        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)
    
    # minority constant classifier
    if args.minority:
        log_param("classifier" , "constant")
        params = {"classifier" : "constant"}
        classifier = DummyClassifier(strategy = "constant", random_state = args.seed, constant = True)
    
    elif args.knn is not None:
        print("    {0} nearest neighbor classifier".format(args.knn))
        log_param("classifier", "knn")
        log_param("k", args.knn)
        params = {"classifier": "knn", "k": args.knn}
        standardizer = StandardScaler()
        knn_classifier = KNeighborsClassifier(args.knn, n_jobs = -1)
        classifier = make_pipeline(standardizer, knn_classifier)
    
    # complement naive bayes classifier
    elif args.complement_naive_bayes:
        print("    complement naive bayes classifier")
        alpha=args.complement_naive_bayes_alpha
        fit_prior=args.complement_naive_bayes_fit_prior
        norm=args.complement_naive_bayes_norm
        print("   Complement Naive Bayes Classifier, alpha = {0}, fit_prior = {1}, norm = {2}".format(alpha, fit_prior, norm))
        log_param("classifier", "cnb")
        log_param("cnb_alpha", alpha)
        log_param("cnb_fit_prior", fit_prior)
        log_param("cnb_norm", norm)
        params = {"classifier": "cnb", 
                  "cnb_alpha" : alpha,
                  "cnb_fit_prior" : fit_prior,
                  "cnb_norm" : norm}
        classifier = ComplementNB(alpha=alpha, fit_prior=fit_prior, norm=norm)
        
    # Decision Tree classifier
    elif args.dtc:
        # set default configuration
        criterion_param = "gini"
        splitter_param = "best"
        max_depth_param = None
        class_weight_param = None
        # determine console args
        if args.dtc_criterion_entropy:
            criterion_param = "entropy"
        if args.dtc_splitter_random:
            splitter_param = "random"
        if args.dtc_max_depth is not None:
            max_depth_param = args.dtc_max_depth
        if args.class_weight_balanced:
            class_weight_param = "balanced"
        print("   Decision Tree Classifier, criterion = {0}, splitter = {1}, max_depth = {2}, class_weight = {3}".format(criterion_param, splitter_param, max_depth_param, class_weight_param))
        #mlflow logging
        log_param("classifier", "decision_tree")
        log_param("dt_criterion", criterion_param)
        log_param("dt_spliiter", splitter_param)
        log_param("dt_max_depth", max_depth_param)
        log_param("dt_class_weight", class_weight_param)
        params = {"classifier" : "decision_tree", "dt_criterion" : criterion_param, 
                  "dt_spliiter" : splitter_param, "dt_max_depth" : max_depth_param,
                  "dt_class_weight" : class_weight_param}
        #classifier 
        classifier = DecisionTreeClassifier(criterion = criterion_param,
                                            splitter = splitter_param,
                                            max_depth = max_depth_param,
                                            class_weight = class_weight_param)
    
    # Random Forest classifier
    elif args.rfc:
        #set default configuration
        criterion_param = "gini"
        bootstrap_param = True
        max_depth_param = None
        n_estimators_param = 100
        class_weight_param = None
        #determine console args
        if args.rfc_criterion_entropy:
            criterion_param = "entropy"
        if args.rfc_no_bootstrap:
            bootstrap_param = False
        if args.rfc_max_depth is not None:
            max_depth_param = args.rfc_max_depth
        if args.rfc_n_estimators != 100:
            n_estimators_param = args.rfc_n_estimators
        if args.class_weight_balanced:
            class_weight_param = "balanced"
        print("   Random Forest Classifier, criterion = {0}, bootstrap = {1}, max_depth = {2}, n_estimator = {3}, class_weight = {4}".format(criterion_param, bootstrap_param, max_depth_param, n_estimators_param, class_weight_param))
        #mlflow logging
        log_param("classifier", "random_forest")
        log_param("rf_criterion", criterion_param)
        log_param("rf_bootstrap", bootstrap_param)
        log_param("rf_max_depth", max_depth_param)
        log_param("rf_n_estimator", n_estimators_param)
        log_param("rf_class_weight", class_weight_param)
        params = {"classifier" : "random_forest", "rf_criterion" : criterion_param, "rf_bootstrap" : bootstrap_param,
                  "rf_max_depth" : max_depth_param, "rf_n_estimator" : n_estimators_param, "rf_class_weight" : class_weight_param}
        #classifier
        classifier = RandomForestClassifier(criterion = criterion_param,
                                            bootstrap = bootstrap_param,
                                            max_depth = max_depth_param,
                                            n_estimators = n_estimators_param,
                                            class_weight = class_weight_param,
                                            #use all processors
                                            n_jobs = -1)

    # Support Vector Classifier
    elif args.svc:
        print("   Support vector classifier")
       
        #param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        #grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        #grid.fit(data["features"], data["labels"].ravel())
        #best_params = grid.best_params_
        kernel = args.svc_kernel
        C = args.svc_c
        gamma = args.svc_gamma
        print("   Support Vector Classifier, c = {0}, gamma = {1}, kernel = {2}".format(C, gamma, kernel))
        log_param("classifier", "svc")
        log_param("kernel", kernel)
        log_param("C", C)
        log_param("gamma", gamma)
        params = {"classifier": "svc", "svc_kernel": kernel, "svc_C": C, "svc_gamma": gamma}
        classifier = SVC(kernel=kernel, C=C, gamma=gamma)

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
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
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
