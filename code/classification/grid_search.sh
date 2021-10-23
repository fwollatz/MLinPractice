#!/bin/bash

mkdir -p data/classification

# specify hyperparameter values
values_of_k=("1 2 3 4 5 6 7 8 9 10")
values_of_dtc_maxdepth=("None 5 10 15 20")
values_of_cnb_alpha=("0.05 0.15 0.30 0.50 0.70 0.85 0.95 1.0")

# different execution modes
if [ $1 = local ]
then
    echo "[local execution]"
    cmd="code/classification/classifier.sge"
elif [ $1 = grid ]
then
    echo "[grid execution]"
    cmd="qsub code/classification/classifier.sge"
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi

# do the grid search 
# KNN
echo "Optimizing KNN Classifier"
for k in $values_of_k
do
    echo "KNN k={$k}"
    $cmd 'data/classification/clf_KNN_'"$k"'.pickle' --knn $k -s 42 --accuracy --kappa --auc
done

#Decision Tree Classifier
echo "Optimizing Decision Tree Classifier"
for depth in $values_of_dtc_maxdepth
do
    echo "DTC max_depth={$depth}"
    $cmd 'data/classification/clf_DTC_'"$depth"'.pickle' --dtc --dtc_max_depth $depth -s 42 --accuracy --kappa --auc
    #running with random splitter
    $cmd 'data/classification/clf_DTC_'"$depth"'_random_splitter.pickle' --dtc --dtc_max_depth $depth --dtc_splitter_random -s 42 --accuracy --kappa --auc
    #running with entropy criterion
    $cmd 'data/classification/clf_DTC_'"$depth"'_criterion_entropy.pickle' --dtc --dtc_max_depth $depth --dtc_criterion_entropy -s 42 --accuracy --kappa --auc
    #running with balanced weight option
    $cmd 'data/classification/clf_DTC_'"$depth"'_class_weight_balanced.pickle' --dtc --dtc_max_depth $depth --class_weight_balanced -s 42 --accuracy --kappa --auc
done

#Random Forest Classifier: Not optimized due to git large file issues

#Naive Bayes Classifier
echo "Optimizing Complement Naive Bayes Classifier"
for alpha in $values_of_cnb_alpha
do
    echo "CNB alpha={$alpha}"
    $cmd 'data/classification/clf_CNB_'"$alpha"'.pickle' --cnb --complement_naive_bayes_alpha $alpha -s 42 --accuracy --kappa --auc
    #running with fit prior
    $cmd 'data/classification/clf_CNB_'"$alpha"'_fit_prior.pickle' --cnb --complement_naive_bayes_alpha $alpha --complement_naive_bayes_fit_prior -s 42 --accuracy --kappa --auc
    #running without normalization
    $cmd 'data/classification/clf_CNB_'"$alpha"'_without_norm.pickle' --cnb --complement_naive_bayes_alpha $alpha --complement_naive_bayes_norm -s 42 --accuracy --kappa --auc
done

#Support Vector Classifier: Hyperparameter Optimizing happening internally
echo "Optimizing Support Vector Classifier"

$cmd 'data/classification/clf_SVC.pickle' --svc -s 42 --accuracy --kappa --auc