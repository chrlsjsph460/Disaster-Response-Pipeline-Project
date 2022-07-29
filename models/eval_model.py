# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 06:32:03 2022

@author: charl
"""
import sys
import time
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import pandas as pd
from helper_functions import load_data, tokenize, build_model, evaluate_model, save_model, train_model, CustomMultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib      

database_filepath = r"..\data\clean_messages.db" 


print('Loading data...\n    DATABASE: {}'.format(database_filepath))
X, Y = load_data(database_filepath)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

model = joblib.load(r"..\models\saved_model.pkl")
print(model.get_params())
print(evaluate_model(model, X_test, Y_test))