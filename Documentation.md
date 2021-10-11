Machine Learning in Practice (Group HumbleBees 777, Qirui Zhu, Frederik Wollatz, Xi Cheng)

**Setup**
1. Setup environment and working tools (incl. Jamovi for dataset exploration) (211004)
2. Defined labeling/color coding convention used on Trello (211004)
3. Explored dataset and listed features that could be used later for model building and experiments (incl. filtering out non-english tweets) (211004)
4. created 'Documentation.md' (211005)
5. clean code: defined naming conventions that shall be used consistently throughout the project (incl. variable names and function names)

##Test Strategy

**Test Agreements**

1. TestCases should named as boolean outcomes (e.g. def this_class_works: => y/n)

2. TestCases should start with "test"

3. Having a seperate test python file per class/components

4. Tests are grouped in a seperate "test" folder

5. Group Tests in 3 steps:
  - I) Arrange: set up of the test => variables, environment (excluding imports)
  - II) Act: perform the task => e.g. testing tokenizer put all tokenizing code in here
  - III) Assert: assertion code in here
	
    
Example:

	```python
    def test_tokenization_single_sentence(self):
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
    
6.) Run related tests before pushing (=> maybe create a shell script)

**Data Loading**

**Preprocessing**
1. 

**Feature Extraction**

**Dimensionality Reduction**

**Classification**
1. added 'topkaccuracy', 'ROC curve'', 'auc' to evaluation metrics in run_*classifer.py* (211005)
2. 

# Documentation Example

Some introductory sentence(s). Data set and task are relatively fixed, so 
probably you don't have much to say about them (unless you modifed them).
If you haven't changed the application much, there's also not much to say about
that.
The following structure thus only covers preprocessing, feature extraction,
dimensionality reduction, classification, and evaluation.

## Evaluation

### Design Decisions

Which evaluation metrics did you use and why? 
Which baselines did you use and why?

### Results

How do the baselines perform with respect to the evaluation metrics?

### Interpretation

Is there anything we can learn from these results?

## Preprocessing

I'm following the "Design Decisions - Results - Interpretation" structure here,
but you can also just use one subheading per preprocessing step to organize
things (depending on what you do, that may be better structured).

### Design Decisions

Which kind of preprocessing steps did you implement? Why are they necessary
and/or useful down the road?

### Results

Maybe show a short example what your preprocessing does.

### Interpretation

Probably, no real interpretation possible, so feel free to leave this section out.

## Feature Extraction

Again, either structure among decision-result-interpretation or based on feature,
up to you.

### Design Decisions

Which features did you implement? What's their motivation and how are they computed?

### Results

Can you say something about how the feature values are distributed? Maybe show some plots?

### Interpretation

Can we already guess which features may be more useful than others?

## Dimensionality Reduction

If you didn't use any because you have only few features, just state that here.
In that case, you can nevertheless apply some dimensionality reduction in order
to analyze how helpful the individual features are during classification

### Design Decisions

Which dimensionality reduction technique(s) did you pick and why?

### Results

Which features were selected / created? Do you have any scores to report?

### Interpretation

Can we somehow make sense of the dimensionality reduction results?
Which features are the most important ones and why may that be the case?

## Classification

### Design Decisions

Which classifier(s) did you use? Which hyperparameter(s) (with their respective
candidate values) did you look at? What were your reasons for this?

### Results

The big finale begins: What are the evaluation results you obtained with your
classifiers in the different setups? Do you overfit or underfit? For the best
selected setup: How well does it generalize to the test set?

### Interpretation

Which hyperparameter settings are how important for the results?
How good are we? Can this be used in practice or are we still too bad?
Anything else we may have learned?
