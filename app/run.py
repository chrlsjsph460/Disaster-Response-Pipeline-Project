import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram, Box
import joblib
from sqlalchemy import create_engine
import sys

# adding models folder to the system path
sys.path.insert(0, r"C:\Users\charl\Desktop\Udacity_DataScience\Disaster-Response-Pipeline-Project\models")


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
db_filepath = r"..\data\clean_messages.db"
engine = create_engine(f'sqlite:///{db_filepath}')
df = pd.read_sql_table('Messages_Clean', engine)

# load model
model = joblib.load(r"..\models\saved_model.pkl")
messages = df.message.apply(tokenize)
message_wordcounts = df.message.apply(len)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categories = df.iloc[:,4:].sum(axis=0).sort_values(ascending = False)
    category_names = categories.index.tolist()
    category_counts = categories.values.tolist()
    
    print(genre_counts)
    print(genre_names)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
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
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Labels',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label"
                }
            }
        },
        {
            'data': [
                Box(
                    y=np.log10(message_wordcounts),
                    marker = dict(color = "#c6e2ff"),
                    boxpoints = "outliers",
                    name = "",
                    jitter = 0.3
                    
                )
            ],

            'layout': {
                'title': 'Words Per Message',
                'yaxis': {
                    'title': "log(Count)"
                },
                'xaxis': {
                    'title': ""
                }
            }
        }
    ]
    # Extract verbs and nouns for some underrepresented categories
    # make pie chart
    # earthquake, water, medical_pro, and death
 
    print(genre_names)
    print(genre_counts)
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

