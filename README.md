[//]: # (Image References)

[image1]: ./images/CleanedDataset_disaster_messages_distr.PNG "Disaster Classification:"
[image2]: ./images/CleanedDataset_disaster_messages_categories_distr.PNG "Category Distribution:"
[image3]: ./images/CleanedDataset_disaster_messages_multipleCategoryLabelsDistribution.PNG "Multiple Label Distribution:"
[image4]: ./images/DisasterMessages_categories_correlationMatrix.PNG "Categories Correlation Matrix:"


# Disaster Messages Classifier Project

## Project Overview
### General
Welcome to this **Natural Language Processing** project from Udacity: we analyse disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for a web app API that categorises this disaster messages.

In an emergency case, the disaster response organisations have to deal with millions of messages, in a situtation they have the least capacity to pull out the messages which are the most important. First, it has to be decided if the message is a real disaster one or not.

![Disaster Classification:][image1]


Different organisations are taking care of different parts of the emergency case (e.g. infrastructure or medical topics, refugee status, ...) and have to know which messages are relevant for them. The given datasets are labelled according this different problem parts, called categories.

After some cleaning steps of the merged dataset the following distribution of the remaining categories is given:

![Category Distribution:][image2]

Regarding the machine learning pipeline, we work on a multi-class, multi-output text classification which assigns to each message sample a set of category target labels.<br>

According scikit-learn [documentation](https://scikit-learn.org/stable/modules/multiclass.html) "In multilabel learning, the joint set of binary classification tasks is expressed with label binary indicator array: each sample is one row of a 2d array of shape (n_samples, n_classes) with binary values: the one, i.e. the non zero elements, corresponds to the subset of labels. An array such as np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]]) represents label 0 in the first sample, labels 1 and 2 in the second sample, and no labels in the third sample." and<br>
"Multioutput classification support can be added to any classifier with MultiOutputClassifier. This strategy consists of fitting one classifier per target. This allows multiple target variable classifications. The purpose of this class is to extend estimators to be able to estimate a series of target functions (f1,f2,f3…,fn) that are trained on a single X predictor matrix to predict a series of responses (y1,y2,y3…,yn)."

The messages are short and an imbalanced data distribution exists. Nevertheless, the messages are mapped to the target categories with a different amount. In general, there are a lot of sparse vectors having mostly 0 values included. Only 1 message is mapped to 27 categories.

![Multiple Label Distribution:][image3]

The correlations of the categories is shown in the correlation matrix.

![Categories Correlation Matrix:][image4]

Correlation values >0.8 are relevant. This fits to the infrastructure features. Around value 0.8 is the feature combination direct_report and request. The category child_alone is empty and therefore a grey column and row appeared. All this shall be handled with the ML pipeline model implementation and not directly with the dataset.

### Information regarding the imbalanced dataset
As we can see, we are dealing with an imbalanced dataset, therefore not all estimator models can be used. One machine learning classifier could be more biased towards the majority class, causing bad classification of the minority class compared to other model types. Therefore we have to take care.

We could do a balancing before classification. The categority classes with low numbers of observations are outnumbered. So, the dataset is highly skewed. To create a balanced dataset several strategies exists:
- Undersampling the majority classes
- Oversampling the minority classes
- Combining over- and under-sampling
- Create ensemble balanced sets

But the goal of this project is not to do associated preprocessing on the dataset (like removing redundant categories or merge redundant information), the goal is the usage of proper feature engineering and model selection.

Another resampling technique is `cross-validation`, a method repeatingly creating additional training samples from the original training dataset to obtain additional fit information from the selected model. It creates an additional model validation set. The prediction model fits on the remaining training set and afterwards is doing its predictions on the validation set. This calculated validation error rate is an estimation of the datasets test error rate. Specific cross validation strategies exist, we are using the `k-fold cross-validation`, that divides the training set in k non-overlapping groups - called folders. One of this folders acts as a validation set and the rest is used for training. This process is repeated k times, each time a different validation set is selected out of the group. The k-fold cross validation estimate is calculated by averaging the single k times estimation results. For k we use 5 because of time consuming calculations and not 10 which would be a better value for k.

According the [paper](https://arxiv.org/ftp/arxiv/papers/1810/1810.11612.pdf) <i>Handling Imbalanced Dataset in Multi-label Text Categorization using Bagging and Adaptive Boosting</i> of 27 October 2018 from Genta Indra Winata and Masayu Leylia Khodra, regarding new data, it is more appropriate to balance the dataset on the algorithm level instead of the data level to avoid overfitting. The algorithm "approach modifies algorithm by adjusting weight or cost of various classes."<br>
So, the `AdaBoostClassifier` is an ensemble method using boosting process to optimise weights. Evaluation showed that this estimator for the <i>MultiOutputClassifier</i> works best compared to the other ones like e.g. <i>RandomForestClassifier</i>. The <i>AdaBoostClassifier</i> is using the <i>DecisionTreeClassifier</i> as its own base estimator. The tree parameters are changed in the parameter grid to improve the imbalanced data situation. Weak learners are boosted to be stronger learners and the results are aggregated at the end.

Additionally for our task, we do some feature engineering.<br>
`feature-selection` approach which can be done after the feature extraction of the `TfidfVectorizer` that is creating [feature vectors](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer).

Additionally, scikit-learn offers the package [feature decomposition](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition) to reduce the complexity of features. With its help a subsampling is added:
- For the sparse matrix delivered from the `TfidfVectorizer` instance we use 3000 most frequent text features, each feature token shall appear at least 2 times and n-gram wording during grid search hyperparameter tuning. The  importance of the token is increased proportionally to the number of appearing in the disaster messages.
- Feature relationship of the sparse matrix is handled with `TruncatedSVD` for latent semantic analysis (LSA). There, a component relationship parameter is evaluated via grid search hyperparameter tuning. Afterwards we have to normalise again.

### Implementation
The implemented project components are:
1. ETL (Extract, Transform, Load) Pipeline
   - Data Wrangling, with import libraries, gather and cleaning datasets
   - Exploratory Data Analysis (EDA), with data exploration including statistics and visualisations
   - Load, to save the cleaned dataset with its disaster messages in an SQL database
   - Note according cleaning: some project requirements are given, so, not all cleaning steps which would happen in real life shall be done; this is necessary to fulfil the project goal about the ML pipeline
   
   All this tasks are part of the Python notebook file and of the file <i>data/process_data.py</i>.

2. ML (Machine Learning) Pipeline
   - During the disaster messages processing, the English text is tokenized, lower cased, lemmatized and contractions are expanded. Spaces, punctuation and English stop words are removed
   - Scikit-learn's pipeline mechanism automates the training workflow of the machine learning classifiers
   - GridSearch cross validation for parameter hypertuning of each classifier without and with LSA decomposition
   - Performance evaluation for the classifiers using specific metrics to find the best model for the NLP task
   
   This Python notebook pipeline work is stored in the file <i>models/train_classifier.py</i> as well. 
   
3. Flask Web App<br>
This web application classifies a newly added disaster text message into the categories to reach an appropriate relief agency for help.<br>
[Flask](https://palletsprojects.com/p/flask/) is a popular Python microframework to build web applications for easy up to more complex tasks. Information how to use it can be found on this [Quickstart](https://flask.palletsprojects.com/en/1.1.x/quickstart/#static-files) page or on Miguel Grinbergs [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world).<br>
For this project, Flask is used together with [Bootstrap 3](https://www.w3schools.com/bootstrap/default.asp).

   - On your local computer to start the web application, in your command line tool change to the <i>app</i> directory of the project and run the following command: 
     ```
     python run.py
     ```
     Some information is shown ending with '* Running on http://0.0.0.0:3001/ (Press CTRL+C to quit)' ...
   - Then on your browser call:
     ```
     http://localhost:3001/
     ```
     to start the web application.

## Project Instructions
This project is implemented with Python 3.7, scikit-learn and scikit-multilearn etc. using its own virtual environment. Python 3 includes already the virtual environment support.<br>
So, working on your local machine, first navigate to the project folder
- cd disaster-messages-project

and then for 
- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):
	```
	conda create --name disaster-messages-project python=3.7
	activate disaster-messages-project
	pip install -r requirements/requirements.txt
	```
- (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `disaster-messages-project` environment:
	```
	python -m ipykernel install --user --name disaster-messages-project --display-name "disaster-messages-project"
	```
- Open the notebooks, e.g. getting the whole list of files by calling:
	```
	jupyter notebook
	```
