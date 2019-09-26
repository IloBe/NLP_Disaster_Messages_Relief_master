[//]: # (Image References)

[image1]: ./images/CleanedDataset_disaster_messages_distr.PNG "Disaster Classification:"
[image2]: ./images/CleanedDataset_disaster_messages_categories_distr.PNG "Category Distribution:"


# Disaster Messages Classifier Project

## Project Overview
### General
Welcome to this **Natural Language Processing** project: we analyse disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for a web app API that categorises this disaster messages.

In an emergency case, the disaster response organisations have to deal with millions of messages, in a situtation they have the least capacity to pull out the messages which are the most important. First, it has to be decided if the message is a real disaster one or not.

![Disaster Classification:][image1]


Different organisations are taking care of different parts of the emergency case (e.g. infrastructure or medical topics, refugee status, ...) and have to know which messages are relevant for them. The given datasets are labelled according this different problem parts, called categories.

After some cleaning steps of the merged dataset the following distribution of the remaining categories is given:

![Category Distribution:][image2]


### Implementation
The implemented project components are:
1. ETL (Extract, Transform, Load) Pipeline
   - Data Wrangling, with import libraries, gather and cleaning datasets
   - Exploratory Data Analysis (EDA), with data exploration including statistics and visualisations
   - Load, to save the cleaned dataset in an SQL database

2. ML (Machine Learning) Pipeline
3. Flask Web App

## Project Instructions

