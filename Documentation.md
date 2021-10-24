Machine Learning in Practice (Group HumbleBees 777, Qirui Zhu, Frederik Wollatz, Xi Cheng)

**Setup**
1. Setup environment and working tools (incl. Jamovi for dataset exploration) (211004)
2. Defined labeling/color coding convention used on Trello (211004)
3. Explored dataset and listed features that could be used later for model building and experiments (incl. filtering out non-english tweets) (211004)
4. created 'Documentation.md' (211005)
5. clean code: defined naming conventions that shall be used consistently throughout the project (incl. variable names and function names)

**Implementation Checklist Preprocessing & Feature Extraction**
1. Update corresponding shell script
2. Write new unit test
3. Update ReadMe.md and maybe Documentation.md
4. Execute corresponding shell script
5. Run `test/run_tests.sh`

##Test Strategy

**Test Agreements**

1. TestCases should named as boolean outcomes (e.g. def this_class_works: => y/n)

2. TestCases should start with "test_" (in order to work with the `run_test.sh`

3. Having a seperate test python file per class/components

4. Tests are grouped in a seperate "test" folder

5. Group Tests in 3 steps:
  - (I) Arrange: set up of the test => variables, environment (excluding imports)
  - (II) Act: perform the task => e.g. testing tokenizer put all tokenizing code in here
  - (III) Assert: assertion code in here
	
    
Example:

```
def test_tokenization_single_sentence_is_working(self):
    #arrange
    input_text = "This is an example sentence"
    output_text = "['This', 'is', 'an', 'example', 'sentence']"
    input_df = pd.DataFrame()
    input_df[self.INPUT_COLUMN] = [input_text]

    #act
    tokenized = self.tokenizer.fit_transform(input_df)

    #assert
    self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text)
```
    
6.) Run `test/run_tests.sh`

## Evaluation

### Design Decisions

Which evaluation metrics did you use and why? 
Which baselines did you use and why?

- Accuracy: to easily compare scores with baseline classifier (majority vote classifier was used here).
- Top K Accuracy score: originally implement for performing multiclass prediction. But now since do bi-class prediction, this metric is not useful anymore.
- Confusion matrix: not a concrete score, but the possibility to get the confusion matrix which some the metrics are based on
- Cohen's Kappa: good performance indication for imbalanced-datasets.
- AUC (Area under the ROC Curve): an aggregate measure of performance across all possible classification thresholds. The higher the score, the better the performance is. 0.5 shows that the classifier does not learn any discriminant features of positive and negative classes while 1 indicates an optimal performance.


### Results

How do the baselines perform with respect to the evaluation metrics?

The baseline classifier was majority vote classifier. On the training set, the accuracy achieved was 0.9058, Cohen's Kappa was 0 and AUC was 0.5. On the validation set, the accuracy achieved was 0.9058, Cohen's Kappa was 0 and AUC was 0.5. Always-true classifier was also implemented.

More details on the classifiers will be illustrated later in the Classifier section. 

All results of the trained classifier could be viewed with:
`mlflow ui --backend-store-uri data/classification/mlflow`

TODO: insert a pic/table of the train/test result 

### Interpretation

Is there anything we can learn from these results?

Accuracy was based on the majority class, which showed the imbalance in the dataset. It was also reflected by Cohen's Kappa. Futhermore, AUC indicated that the classifier has no discrimination capacity to distinguish between positive and negative classes.

The classifier that delivered the best results was decision tree, with the balanced class weight option, gini criterion, max depth of 20 and best splitter. The results were detailed in the following table.


Cohen's Kappa was still pretty low on the validation set while high on the training set for the best classifier. It indicates overfitting of this classifier. Increasing the max depth led to stronger overfitting. The AUC score also confirmed the observation.

## Preprocessing

### Design Decisions

As for the general/classic preprocessing, we decided on well-known NLP preprocessing steps to remove redundancy, 
data that does not convey useful information (e.g. stop words), and tweet specific contents (like urls and emoji) to
preprocess the tweet text itself into plain textualized information. The general preprocessing includes lower casing, 
language filtering, stop word removal, punctuation removal (already given), emoji and url filtering and tokenization.
All intermediate results are stored in separate columns and are thus a accessible for the feature extraction. 

In addition to preprocessing the tweet text itself, for each tweets the emojis were extracted for further
feature extraction.

### Results

Example Preprocessing of one tweet:

Before Preprocessing:

`"Courses@CRG: Containers &amp; #Nextflow  Slow-paced hands-on course designed for absolute beginners who want to start using #containers and @nextflowio pipelines to achieve #reproducibility of data analysis #VEISris3cat #FEDERrecerca #Docker #CloudComputing  ➡️ https://t.co/HxbfIdZVyl  https://t.co/1kGRujM5vB""Courses@CRG: Containers &amp; #Nextflow  Slow-paced hands-on course designed for absolute beginners who want to start using #containers and @nextflowio pipelines to achieve #reproducibility of data analysis #VEISris3cat #FEDERrecerca #Docker #CloudComputing  ➡️ https://t.co/HxbfIdZVyl  https://t.co/1kGRujM5vB"`

After Preprocessing:

`"['coursescrg', 'contain', 'amp', 'nextflow', 'slowpac', 'handson', 'cours', 'design', 'absolut', 'beginn', 'want', 'start', 'use', 'contain', 'nextflowio', 'pipelin', 'achiev', 'reproduc', 'data', 'analysi', 'veisris3cat', 'federrecerca', 'docker', 'cloudcomput']"`

Emojis equals `[➡]`. 

## Feature Extraction

### Design Decisions

Which features did you implement? What's their motivation and how are they computed?

The features implemented are:
- the most common hastags (boolean): determine the n (specifyable) most common hashtags and checks for the existence of these words in the tweet. This creates n new features for each word.
- the most common emojis (boolean): determine the n (specifyable) most common emojis and checks for the existence of these emojis in the tweet. This creates n new features for each emoji.
- character length (implemented previously, int): the length of the unpreprocessed tweet. 
- the amount of followers (int): make use of the twitter API to get the follower count of the person which the tweets belongs to.
- the number of urls (int): determines how many urls were shared in the tweet. As it brings new information to the followers, it has the potential of being retweeted in the social circle. 
- the most common words (str): certain popular or catchy words could attract more views and therefore retweets.
- the number of hashtags (int): the more hashtags, the more likely that a tweet gets identified by the search function. It therefore increases the chance of being discovered by the community and being retweeted.
- if any photos were attached: photos as visual stimuli could play an important role in catching people's attention and in attracting others to spend more time on the tweets.
- the number of words (int): directly influences how much content the person could send out in one tweet. A high number indicates more content, while a small number indicates little content
- the unix time when tweets were published (int):the unix-time when the tweet was published serves as indicator when in the complete timeline of all these tweets, it was posted. Thereby it might be possible for the classifier to find trending topics/hashtags/etc.
- the month, weekday, hour (one-hot-encoded booleans): the *month* that the post was send in, could be related to how many people see the tweet. In December or the summer months, when people usually go on vacation, might lead to more people on twitter in general, because of more spare time.
The *weekday*, similar to the *month*, with people having more time to read through twitter on the weekend. The *hour of the day* is grouped in blocks of 3 hours. It could show tendencies of people using plattforms like twitter more commonly in the afternoon/evening, than during typical working-hours 
- The sentiment: indicated by the vader-module by nltk, shows if the language is more positive or more negative. This could result in a more positive/negative mindset of the reader, and therefore in a higher chance of getting shared.

### Results

Can you say something about how the feature values are distributed? Maybe show some plots?

TODO: 
- pick and display numbers of the most common ones
- export CSV file (after preprocessing)
- add some plots of the extracted features 

### Interpretation

Can we already guess which features may be more useful than others?

The following features may be more useful: the amount of followers. 

The above features were chosen because we believed that they would play a role collectively in determining the virality of tweets. And we expected the amount of followers to be more important. 

## Dimensionality Reduction

If you didn't use any because you have only few features, just state that here.
In that case, you can nevertheless apply some dimensionality reduction in order
to analyze how helpful the individual features are during classification

### Design Decisions

Which dimensionality reduction technique(s) did you pick and why?

It was originally decided to choose three dimensionality reduction methods. And the following dimensionality reduction techniques were used:
- PCA: used to perform feature projection and to compute new features based on the original ones; this approach select a number of principle components automatically based on the accumulative explained variance ratio (currently 0.95).
- Wrapper (RFE) method: used for feature selection based on the models that evaluated different features; the number of selectable features could be specified.
- Filter method: instead of using models, the mutual information heuristics were used to select suitable features; the number of selectable features could be specified.

### Results

Which features were selected / created? Do you have any scores to report?

TODO: insert a pic of 'RFE reducer with n = 10, model = DTC'.
Title of pic: example of feature selection using wrapper method with top ten features and the decision tree model.

RFE was used as dimensionality reduction before hyper-parameter optimization.

TODO: add PCA scores and explanation

### Interpretation

Can we somehow make sense of the dimensionality reduction results?
Which features are the most important ones and why may that be the case?

Dimensionality reduction indicated that the amount of followers was not so useful as thought.

TODO: provide a short explanation why the amount of followers is not important 
TODO: qirui: PCA interpretation

## Classification

### Design Decisions

Which classifier(s) did you use? Which hyperparameter(s) (with their respective
candidate values) did you look at? What were your reasons for this?

The following classifiers were used:
- majority vote classifier: parameters - seed
- label frequency classifier: parameters - seed
- minority vote classifier: parameters - seed
- k-nearest neighbor classifier: parameters - *k*
- complement naive bayes: parameters - alpha, fit_prior, norm
    - it was chosen because it is known to work with an imbalanced dataset like the current dataset
- decision tree classifier: parameters - criterion, splitter, max_depth, class_weight
- random forest classifier: parameters - criterion, bootstrap, max_depth, n_estimators, class_weight
- support vector classifier: paramters - c, gamma, kernel

TODO: individual filling in

TODO: update minority vote classifier cmd in readme

### Results

The big finale begins: What are the evaluation results you obtained with your
classifiers in the different setups? Do you overfit or underfit? For the best
selected setup: How well does it generalize to the test set?

The performance of random forest classifier and support vector classifier could not be optimized because the first one was too large to be pushed to git while the latter took too long for classification. For the remaining classifiers, the decision classifier performs the best, followed by the k-nearest neighbor classifier, complement naive bayes classifier, majority-/minority-vote and label frequency classifier in this order. The last three classifiers were of similar performance. Details were summarized in the mlflow table. 

In general, nearly all the classifiers were overfitting the training set. The decision tree classifier generailzed better compared to other classifier, even though it could be optimized further. 

TODO: add config of the best model 

### Interpretation

Which hyperparameter settings are used? How important for the results?
How good are we? Can this be used in practice or are we still too bad?
Anything else we may have learned?

TODO: qirui - add hyper-parameter values

We suspected that the decision tree classifier performed the best because the dimensionality reduction was performed using a decision tree model.
The best classifier managed to score at around 80% on the validation set but still the Cohen's Kappa score of 0.292 indicated that overfitting persisted and further optimization was needed.

TODO: qirui - try hyper-parameter on the test set

The following observations were made during the experiments:
- adjusting alpha value did not impact the performance of cnb classifier
- as cnb does not take negative values, pca could not be applied beforehand for this classifier

TODO: individuals add observations 