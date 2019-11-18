
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from langdetect import detect, DetectorFactory


def load_data(messages_filepath, categories_filepath):
    '''
    Loads csv files and returns its merge result as dataframe
    
    Input:
      messages_filepath: disaster messages csv file 
      categories_filepath: disaster category csv file
    Output:
      df: the combined dataframe of messages and categories
    '''

    # load csv messages and category datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='inner', on=['id'])
        
    return df



def select_language_codes(df_source):
    '''
    Maps original text language code info to each dataframe message,
    e.g. en for English

    Input:
      df_source: the dataframe the original languages shall be known
    Output:
      languages: the list of all found text messages language codes
    '''

    # see: https://pypi.org/project/langdetect/
    # Language detection algorithm is non-deterministic,
    # which means that if you try to run it on a text which is either
    # too short or too ambiguous, you might get different results everytime you run it.
    # Therefore the DetectorFactory.seed is necessary.

    DetectorFactory.seed = 14
    message_subset = df_source['message']
    languages = []
    for message in message_subset.values:
        try:
            if message in (None, ''):
                lang_code = None
            else:
                lang_code = detect(message)
        except:
            lang_code = None
        languages.append(lang_code)
        #print("message: {} - lang_code: {}".format(message, lang_code))
        
    return languages


def modify_categories(df):
    '''
    Maps the category types to one-hot encoded feature columns

    Input:
      df: dataframe with category column information
    Output:
      df: modified dataframe having one column for each category type
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=";", expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1).values
    # use this row to extract a list of new column names for categories
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = []
    # the row variable itself has length 1, but has 36 category elements
    for i in range(len(row[0])):
        s = pd.Series([row[0][i]])
        colname = s.str.split(pat='-', n=1)
        category_colnames.append(colname[0][0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        # convert column from string to numeric
        categories[column] =  categories[column].str.split("-").str[1].astype('int64')

    # regarding the column 'related', there are messages with value 2 which is probably a mistake,
    # we change it to be 0 instead of 2.
    categories['related'].replace([2], 0, inplace=True)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index=True, right_index=True)

    return df


def clean_data(df):
    '''
    Modifies the dataframe to get one-hot encoded category types,
    to get the English disaster messages only
    and finally removes the duplicate rows.

    Input:
      df_source: the merged dataframe including all uncleaned information
    Output:
      df: the cleaned dataframe with English disaster messages only
    '''

    # first find all language codes of messages and add this info to the df
    languages = select_language_codes(df)
    df['lang_code'] = languages

    # create a dataframe with one column for each category label/type
    df = modify_categories(df)

    # drop duplicates
    df.drop_duplicates(inplace=True)


    # use only observations with English messages,
    # after merge the specific id column is useless
    df_en = df.query('lang_code == "en"')
    df_en = df_en.drop(columns=['id'])
    
    return df_en


def save_data(df, database_filename):
    '''
    Saves the dataframe into a sqlite database 

    Input:
        df: cleaned dataframe with messages and categories information
        database_filename: SQL database file name
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df_en.to_sql('Messages_Categories_table', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    