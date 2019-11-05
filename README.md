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

Regarding the machine learning pipeline, we work on a multi-output, multi-label text classification which assigns to each message sample a set of category target labels.<br>
According scikit-learn [documentation](https://scikit-learn.org/stable/modules/multiclass.html) "In multilabel learning, the joint set of binary classification tasks is expressed with label binary indicator array: each sample is one row of a 2d array of shape (n_samples, n_classes) with binary values: the one, i.e. the non zero elements, corresponds to the subset of labels. An array such as np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]]) represents label 0 in the first sample, labels 1 and 2 in the second sample, and no labels in the third sample." and<br>
"Multioutput classification support can be added to any classifier with MultiOutputClassifier. This strategy consists of fitting one classifier per target. This allows multiple target variable classifications. The purpose of this class is to extend estimators to be able to estimate a series of target functions (f1,f2,f3…,fn) that are trained on a single X predictor matrix to predict a series of responses (y1,y2,y3…,yn)."

The messages are short and an imbalanced data distribution exists.

![Multiple Label Distribution:][image3]

The correlations of the categories is shown in the correlation matrix.

![Categories Correlation Matrix:][image4]

Correlation values >0.8 are relevant. This fits to the infrastructure features. Around value 0.8 is the feature combination direct_report and request. The category child_alone is empty and therefore a grey column and row appeared. All this shall be handled with the ML pipeline model implementation and not directly with the dataset.

### Implementation
The implemented project components are:
1. ETL (Extract, Transform, Load) Pipeline
   - Data Wrangling, with import libraries, gather and cleaning datasets
   - Exploratory Data Analysis (EDA), with data exploration including statistics and visualisations
   - Load, to save the cleaned dataset with its disaster messages in an SQL database
   - Note according cleaning: some project requirements are given, so, not all cleaning steps which would happen in real life shall be done; this is necessary to fulfil the project goal

2. ML (Machine Learning) Pipeline
   - During the disaster messages processing, the English text is tokenized, lower cased, lemmatized and contractions are expanded. Spaces, punctuation and English stop words are removed
   - Scikit-learn's pipeline mechanism automates the training workflow of the machine learning classifiers
   - GridSearch cross validation for parameter hypertuning of each classifier
   - Performance evaluation for the classifiers using specific metrics to find the best model for the NLP task
   
3. Flask Web App

## Project Instructions

