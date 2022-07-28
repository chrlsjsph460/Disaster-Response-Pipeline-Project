import pickle
import sys

#Natural Language 
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer

#Machine Learning Models and transformers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

#Metrics and Model Selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score


import pandas as pd
import numpy as np
import sqlalchemy as db



import time

hamming_scorer = make_scorer(hamming_loss, greater_is_better = False)

# Can be used without MultiOutputClassifier
class CustomKnnClassifier(BaseEstimator, ClassifierMixin):
    ''' assume pandas df for training.
        Remove column with 0 variance in training.
        Predict the value seen in training for that category 
    '''
    
    
    def __init__(self, n_neighbors = 5, weights = 'uniform', n_jobs = 1 ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.n_jobs = n_jobs
        
        self.clf = KNeighborsClassifier(n_neighbors= self.n_neighbors, weights= self.weights, n_jobs=self.n_jobs)
        
    def fit(self, X, Y):
        #get columns to drop and values to predict
        self.drop_predict = []
        self.drop_columns = []
        self.drop_index = []
        
        
        index = 0
        for column in Y.columns.to_list():
            if np.var(Y[column]) == 0: 
                self.drop_columns.append(column)
                self.drop_predict.append(Y[column].iloc[0])
                self.drop_index.append(index)
            index += 1
        print(f"dropped columns: {self.drop_columns}")       
        if len(self.drop_columns) > 0:
            Y = Y.drop(self.drop_columns,axis = 1)
            print(f"dropped columns: {self.drop_columns}") 
            self.clf.fit(X,Y)
        else:
            self.clf.fit(X,Y)
        
        return self
    

    def predict(self, X):
        Ypred = self.clf.predict(X)
        m, n = Ypred.shape
        if len(self.drop_columns) > 0:
            #insert prediction for 0 variance column
            col = np.ones((m,1))*self.drop_predict
            Ypred = np.insert(Ypred, self.drop_index, col, axis = 1)

        return Ypred
    
    def predict_proba(self, X):
        probs = self.clf.predict_proba(X)
        m, n = probs.shape
        if len(self.drop_columns) > 0:
            col = np.ones((m, 2))
            col[:,self.drop_predict] = 1
            probs = np.insert(probs, self.drop_index, col, axis = 1)
    
        return probs
        
        
class CustomMultiOutputClassifier(BaseEstimator, ClassifierMixin):
    ''' Applies MultiOutputClassifier to the classifier you feed it
        Didn't bother with validation tests
        assume pandas df for training.
        Remove column with 0 variance in training.
        Predict the value seen in training for that category 
        
        
        
    '''
    
    
    def __init__(self, model = MultinomialNB(), **kwargs):
        self.model = model
        self.model.set_params(**kwargs)
        self.clf = MultiOutputClassifier(self.model)
        
    def fit(self, X, Y):
        #get columns to drop and values to predict
        self.drop_predict = []
        self.drop_columns = []
        self.drop_index = []
        
        
        index = 0
        for column in Y.columns.to_list():
            if np.var(Y[column]) == 0: 
                self.drop_columns.append(column)
                self.drop_predict.append(Y[column].iloc[0])
                self.drop_index.append(index)
            index += 1
        if len(self.drop_columns) > 0:
            Y = Y.drop(self.drop_columns,axis = 1)
            # print(f"dropped columns: {self.drop_columns}") 
            self.clf.fit(X,Y)
        else:
            self.clf.fit(X,Y)
        
        return self
    

    def predict(self, X):
        Ypred = self.clf.predict(X)
        m, n = Ypred.shape
        if len(self.drop_columns) > 0:
            #insert prediction for 0 variance column
            col = np.ones((m,1))*self.drop_predict
            Ypred = np.insert(Ypred, self.drop_index, col, axis = 1)

        return Ypred
    
    def predict_proba(self, X):
        probs = self.clf.predict_proba(X)
        m, n = probs.shape
        if len(self.drop_columns) > 0:
            col = np.ones((m, 2))
            col[:,self.drop_predict] = 1
            probs = np.insert(probs, self.drop_index, col, axis = 1)
    
        return probs


def load_data(database_filepath):
    """
    Parameters
    ----------
    database_filepath : Address of the database storing the disaster-response data
    Returns
    -------
    X : predictors as text messages
    Y : categories text messages were classified to

    """
    engine = db.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("Messages_Clean", engine)
    X = df.message
    Y = df.iloc[:,4:]

    return X, Y


def tokenize(text):
    '''
    Parameters
    -----------
    text : string of data
    
    Returns
    -----------
    words: list of words from text. non-alphanumerics are removed and string 
    is split into words. Words lemmatized using WordNetLemmatizer.
    
    '''
    
    #change to lower case and remove non alpha numeric characters
    text = re.sub(r"[^a-z0-9]", " ", text.lower())
    #split message into words
    words = word_tokenize(text)
    #remove stop words
    words = [word for word in words if word not in set(stopwords.words("english"))]
    #reduce words to their root form
    words =[WordNetLemmatizer().lemmatize(word) for word in words]
    return words



def multiOutputFscore(Ytrue, Ypred):
    _, n = Ytrue.shape
    scores = np.array([f1_score(Ytrue[:,i], Ypred[:,i], zero_division = 0) for i in range(n)])
    return scores

def multiOutputPrecision(Ytrue, Ypred):
    _, n = Ytrue.shape
    scores = np.array([precision_score(Ytrue[:,i], Ypred[:,i], zero_division = 0) for i in range(n)])
    return scores


def multiOutputRecall(Ytrue, Ypred):
    _, n = Ytrue.shape
    scores = np.array([recall_score(Ytrue[:,i], Ypred[:,i], zero_division = 0) for i in range(n)])
    return scores

def rubricScores(Ytrue, Ypred):
    _, n = Ytrue.shape
    scoresF = multiOutputFscore(Ytrue.values, Ypred)
    scoresP = multiOutputPrecision(Ytrue.values, Ypred)
    scoresR = multiOutputRecall(Ytrue.values, Ypred)
    
    scoresDict = {"F1":scoresF, "Precision":scoresP, "Recall":scoresR}
    result = pd.DataFrame(data=scoresDict, index = Ytrue.columns)
    return result


def meanFscore(Ytrue, Ypred):
    
    scores = multiOutputFscore(Ytrue, Ypred)
    return scores.mean()

def meanPrecision(Ytrue, Ypred):
    
    scores = multiOutputPrecision(Ytrue, Ypred)
    return scores.mean()


def meanRecall(Ytrue, Ypred):
   
    scores = multiOutputRecall(Ytrue, Ypred)
    return scores.mean()
    
F1_scorer = make_scorer(meanFscore, greater_is_better = True)
Precision_scorer = make_scorer(meanPrecision, greater_is_better = True)
Recall_scorer = make_scorer(meanRecall, greater_is_better = True)
hamming_scorer = make_scorer(hamming_loss, greater_is_better = False)



def build_model():
    '''
    Input
    -------
    Nothing:

    Returns
    -------
    model_pipeline : Gridsearch cv object from pipeline
     
    Pipeline
    ---------    
    Create pipeline for preprocessing text data
    Use ensemble method on 4 modelds
    Use gridSearchCV to find optimal models later 
    
    Tuning Parameters
    ------------------
    Logistic Regression:     C  (float>0)-> Inverse of regularization parameter. Smaller Cs mean more regularization
    KNN:                     n_neighbors -> the number of nearest neighbors 
                             weights     -> Uniform or based on distance (Nearer points have more influence)
    Random Forest:
    MultinomialNB:           alpha       -> Smoothing parameter. By default, there are alpha for each class


    ''' 
    #Create Multinomial NB instance for pipeline. It's a fast model. 
    # model = CustomMultiOutputClassifier(MultinomialNB())
    model = CustomMultiOutputClassifier(LogisticRegression(class_weight = 'balanced', solver = "newton-cg"))
    # create scoring function
    #Use f1 score to assess model usefulness. This is a balance of precision and recall_score
    
    ############ Other Scorers ###############################################
    # F1_scorer = make_scorer(meanFscore, greater_is_better = True)
    # Precision_scorer = make_scorer(meanPrecision, greater_is_better = True)
    # Recall_scorer = make_scorer(meanRecall, greater_is_better = True)
    ##########################################################################
    hamming_scorer = make_scorer(hamming_loss, greater_is_better = False)
    scoring = {"Hamming": hamming_scorer}
    # scoring = {"F1":F1_scorer}
    
    # Create innerPipeline and gridsearch. This is for the ml model's parameters. 
    # inner pipeline and gridsearch
    innerPipeline = Pipeline([('tfidf',TfidfTransformer()),('clf', model)])     
    # innerParameters = {'clf__model__alpha':np.linspace(0.01, 0.50, 50)}   
    innerParameters = {'clf__model__C': np.logspace(-3.0, 3.0, num=50) } 
    innerScoring = {"F1": F1_scorer}
    # innerScoring = {"Hamming":hamming_scorer}
    # innerSearch = GridSearchCV(innerPipeline, innerParameters, scoring = innerScoring, return_train_score=True, refit = "Hamming", n_jobs = 3, verbose = 0)
    innerSearch = GridSearchCV(innerPipeline, innerParameters, scoring = innerScoring, return_train_score=True, refit = "F1", n_jobs = 3, verbose = 0)


    # outer pipeline and gridsearch
    outerPipeline = Pipeline([('vect',CountVectorizer(tokenizer = tokenize)),('innerCV', innerSearch)])       
    
    return outerPipeline


def evaluate_model(model, X_test, Y_test):
    '''
    Print classification report from model on test data    

    Parameters
    ----------
    model : pipeline object fit to the training data
    X_test : testing predictors of the data set
    Y_test : testing response of the data set

    Returns
    -------
    None.

    '''
    
    return rubricScores(Y_test, model.predict(X_test))


def train_model(X, Y, model):
    '''

    Parameters
    ----------
    X : training features
    y : training targets
    model :  Pipeline object from build model. There is an inner gridsearch. 

    Returns
    -------
    model : best estimator from GridSearchCV object.

    '''
    # Perform train-test split. 20% is reserved for testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    # cross-validation parameters for count vectorizer
    # gridsearch is ran on tfidf and model only. It doesn't make sense to 
    # fit a new countvectorizer each time. It's slow and unnecessary. The most 
    # cross validation fold will learn that it shouldn't is the vocabulary. 
    # This seems reasonable. If words don't show up in any of the fold examples they will be 0. 
    
    
    outerParameters = []
    for gram in [(1,2)]:
        for mind in [5]: #[5,6,7,8]:
            for maxd in [.80]:#, .85, 0.90]:
                outerParameters.append({'vect__ngram_range': gram, 'vect__min_df': mind, 'vect__max_df':maxd})
            
    best_score = -np.inf
    for param in outerParameters:
        start = time.time()
        model.set_params(**param)
        model.fit(X_train, Y_train)
        score = model[1].best_score_ 
        if score > best_score:
            print(f"Improvement!!! \n New model is better. new score: {score} > old score: {best_score}.")
            model_ = Pipeline([('vect',model[0]),
                              ('innerPipeline', model[1].best_estimator_)])
            best_score = score
            print(f"CountVectorizer Vocabulary length: {len(model_[0].vocabulary_)}")
            print(f"Tfidf number of features: {model_[1][0].n_features_in_}")

        stop = time.time()
        print(f"time taken: {stop - start}")
    

    return model_


        
        
def save_model(model, model_filepath):
    '''

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    model_filepath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)
    return None


