import sys
import time
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import pandas as pd
from helper_functions import load_data, tokenize, build_model, evaluate_model, save_model, train_model, CustomMultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
      

def main():
    for arg in sys.argv:
        print(arg)
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start = time.time()
        model = train_model(X, Y, model)
        end = time.time()
        print(f"Training took {(end-start)/3600} hours....")
        

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
