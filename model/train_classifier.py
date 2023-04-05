import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
import sqlite3
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    '''Load database from the filebath and save as sql lite DB'''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message']
    y = df.loc[:, 'related':'direct_report']
    y=y.astype(int)
    category_names=y.columns.tolist()
    return X, y, category_names

def tokenize(text):
    '''Tokenize text by words and lemmatize the tokens, then return clean tokens'''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''Build a random forest classifier with multioutput, using optimal parameters found through GridSearchCV'''
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Creating parameters
    parameters = {'clf__estimator__min_samples_split': [2, 3, 4]
    }
    # Creating GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    '''Run classification report for each category which outputs f1 score, precision and recall'''
    y_pred = model.predict(X_test)
    return (classification_report(y_test, pd.DataFrame(y_pred, columns=y_test.columns), target_names=category_names))

def save_model(model, model_filepath):
    '''Saves the final model as a .pkl file'''
    filename = model_filepath
    return pickle.dump(model, open(filename, 'wb'))

def main():
    '''Loads data then builds, fits and evaluates a model and then saves the trained model.'''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
