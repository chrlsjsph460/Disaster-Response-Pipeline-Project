import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
import pandas as pd
import numpy as np
import sqlalchemy as db
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(database_filepath):
    engine = db.create_engine(database_filepath)
    df = pd.read_sql_table("Messages_Clean", engine)
    X = df.messages
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    #change to lower case and remove non alpha numeric characters
    text = re.sub(r"[^a-z0-9]", " ", text.lower())
    #split message into words
    words = word_tokenize(text)
    #remove stop words
    words = [word for word in words if word not in set(stopwords.words("english"))]
    #use stemmer
    stemmer = SnowballStemmer(language="english")
    words = [stemmer.stem(word) for word in words]
    return words


def build_model():
    pipeline = Pipeline([('count_vect',CountVectorizer(tokenizer=tokenize)), 
                    ('tfidf',TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    col_names = Y_test.columns 
    y_pred = model.predict(X_test)
    i = 0
    y_pred_df = pd.DataFrame()
    for name in col_names:
        y_pred_df[name] = y_pred[:,i]
        i += 1
        report = classification_report(Y_test[name],y_pred_df[name],output_dict = True)
        print(report)


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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()