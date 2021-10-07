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