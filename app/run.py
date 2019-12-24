import json
import plotly
import random as rn
import numpy as np
import pandas as pd
import string
import pickle
import collections
from collections import Counter

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.remove('no')
stop_words.remove('not')

from langdetect import detect, DetectorFactory
from bs4 import BeautifulSoup

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# deprecated in scikit-learn 0.21, removed in 0.23
#from sklearn.externals import joblib   
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}


# see: https://pypi.org/project/langdetect/
# Language detection algorithm is non-deterministic,
# which means that if you try to run it on a text which is either
# too short or too ambiguous, you might get different results everytime you run it.
# Therefore the DetectorFactory.seed is necessary.
def lang_detect(text):
    '''
    Detects the language of the input text reflections
    
    Input
        a string text
    Output
        a list of detected languages
    '''
    DetectorFactory.seed = 14
    
    lang = []
    for refl in text:
        lang.append(detect(refl))
    return lang


def check_word_en(text):
    '''
    Checks if the word is an English one by usage of the WordNet vocabulary
    
    Input
        text string to check if being part of the English vocabulary
    Output
        returns the remaining English word list if available and informs
        the user if non English words or strings are available
    '''
    
    text_str = []
    for word in text.split():
        # Check to see if the words are in the dictionary
        if wn.synsets(word):
            text_str.append(word)
        else:
            if lang_detect(word) != 'en':
                message = "The text part '" + word + "' is not an English word. Change your input message."
                print(message)        
    return text_str


# function from Dipanjan's repository:
# https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%\
# 20content/nlp%20proven%20approach/NLP%20Strategy%20I%20-%20Processing%20and%20Understanding%20Text.ipynb
def expand_contractions(text, contraction_mapping):
    '''
    Expands shortened text parts included in the disaster text message
    
    Input
        text: text message that shall be controlled of having shortened text
        contraction_mapping: dictionary with shortened key text and their long text version value
    Output
        expanded_text
    '''
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    
    return expanded_text


def tokenize(text):
    '''
    Tokenises the given text data
    
    Input
        text: the new disaster text message
    Output
        clean_tokens: list of cleaned tokens, means English words which are normalised, contracted,
        tokenised, lemmatised and removed from English stop words
    '''
    
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'       
    detected_urls = re.findall(url_regex, bom_removed)
    for url in detected_urls:
        text = bom_removed.replace(url, "urlplaceholder")
        
    # change the negation wordings like don't to do not, won't to will not 
    # or other contractions like I'd to I would, I'll to I will etc. via dictionary
    text = expand_contractions(text, CONTRACTION_MAP)

    # remove punctuation [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]
    text = text.translate(str.maketrans('','', string.punctuation))
    # remove numbers
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    # during ETL pipeline we have reduced the dataset on English messages ('en' language coding,
    # but there can be some wrong codings    
    tokens = word_tokenize(letters_only, language='english')
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # remove stop words and take care of words having at least 2 characters
        if (len(clean_tok) > 1) & (clean_tok not in stop_words):
            clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/Disaster_Messages_engine.db')
df = pd.read_sql_table('Messages_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    for text in df['message'].values:
        tokenized_ = tokenize(text)
    
    # extract data needed for visuals
    # Genre message distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Category message distribution
    df_related_1 = df.query("related == 1")
    dict_cat = df_related_1.iloc[:, 5:].sum().to_dict()
    sorted_feature = sorted(dict_cat.items(), key=lambda kv: kv[1])
    dict_sorted = collections.OrderedDict(sorted_feature)
    labels = list(dict_sorted.keys())
    values = list(dict_sorted.values())
    
    # create visuals
    # shows genre and category distribution graphs
    figures = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Message Distribution by Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=labels,
                    y=values
                )
            ],

            'layout': {
                'title': 'Messages Distribution by Category',
                'yaxis': {
                    'title': "Count",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45,
                    'automargin':True
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query (note: this is the text query part of the master.html)
    query = request.args.get('query', '') 
    print("Query")
    print(query)
    
    # creates word tokens out of the query string
    query_tokens = tokenize(query)
    print("Tokenized query text:")
    print(query_tokens)
    text = ' '.join(query_tokens)
    
    # checks if the words are are English words,
    # if not remove it from the query tokens and inform the user
    modified_query_list = check_word_en(text)
    modified_query = ' '.join(modified_query_list)
    print("Modified query text:")
    print(modified_query)

    # use model to predict classification for query
    classification_labels = model.predict([modified_query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query = "The original message text is: '" + query + 
                "' and the modified English query words are: '" + modified_query + "'",
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
    
