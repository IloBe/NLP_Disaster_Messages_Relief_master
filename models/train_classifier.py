# import libraries
import sys
import random as rn
import numpy as np
import pandas as pd
import string
import pickle
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
stop_words = set(stopwords.words('english'))

from bs4 import BeautifulSoup

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# warnings status to show
import warnings
warnings.warn("once")

# reproducible
FIXED_SEED = 42
np.random.seed(FIXED_SEED)
rn.seed(FIXED_SEED)


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


def load_data(database_filepath):
    '''
    Load the SQL database into a dataframe
    
    Input:
        database_filepath: database file to be read
    Output:
        X: input (X) messages samples
        y: output (y) samples, the categories are the targets of the classification task
        category_names: the set of category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages_Categories_table', engine)
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    
    return X, y, category_names


def expand_contractions(text, contraction_mapping):
    '''
    Contract the text message, if necessary
    
    Input:
        text:
        contraction_mapping: 
    Output:
        expanded_text: the modified expanded text message
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
    soup = BeautifulSoup(text, 'html')
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
    tokens = word_tokenize(letters_only, language='english')
    # for the lexical correctly found word stem (root)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # use only lower cases, remove leading and ending spaces
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # Small ones are probably no relevant words ...
        # remove English stop words
        if (len(clean_tok) > 1) & (clean_tok not in stop_words):
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    ''' 
    Build the model pipeline for the estimator model used for the MultiOutputClassifier,
	it uses AdaBoostClassifier as estimator with its own DecisionTreeClassifier basic estimator
	and balanced class weights
    
    Output:
        pipeline: returns the created machine learning model pipeline
    ''' 
    
    pipeline = Pipeline([
        ('features', FeatureUnion([            
            ('text_pipeline', Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=tokenize, sublinear_tf=True,
										  ngram_range: (1, 3),
                                          max_features=3000, min_df=2)),
                ('best', TruncatedSVD(n_components=100, random_state=FIXED_SEED)),
                ('normalizer', Normalizer(copy=False))
                
            ]))           
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=100, class_weight='balanced',
                                                         n_jobs=1, random_state=FIXED_SEED)))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Text summary of the overall accuracy and precision, recall, F1 score for each target class
    
    Input:
        model:
        X_test: independent features of the test dataset
        Y_test: dependent features of the test dataset
        category_names: names of the dependent target class features
    Output:
        Overall Accuracy and Classification Report: print F1_score, precision and recall for each target class 
    '''
    # prediction calculation
    y_pred = model.predict(X_test)
    
    # accuracy
    print("\nFirst: overall accuracy score: {:5f}".format(accuracy_score(y_test, y_pred)))

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    # shows F1_score, precision and recall for each target class
    class_report = classification_report(Y_test, y_pred, target_names=category_names)
    print("Classification Report for each target class:\n", class_report)


def save_model(model, model_filepath):
    '''
    Save model as a pickle file
    
    Input:
        model:
        model_filepath:
    '''
    pickle.dump(model, open(model_filepath, "wb" ))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/Disaster_Messages_engine.db classifier.pkl')


if __name__ == '__main__':
    main()