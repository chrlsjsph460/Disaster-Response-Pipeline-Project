import sys
import pandas as pd
import sqlalchemy as db
import matplotlib.pyplot as plt
import seaborn as sns
 

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how="left", on="id")
    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=";", expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    row = row.str.split("-")

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x[0] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [row[-1] for row in categories[column]]  
        # convert column from string to numeric
        categories[column] = [int(row) for row in categories[column]]
    # convert 2s in related column to 0 
    categories.related = [row if row ==1 else 0 for row in categories.related]
    
    # drop the original categories column from `df`
    df.drop(["categories"], axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    # check number of duplicates
    duplicates = df.duplicated(keep = "first")
    # drop duplicates
    df_clean = df.loc[~duplicates]
    # check number of duplicates
    duplicate_rows = sum(df_clean.duplicated())
    print("There are {} non-unique rows".format(duplicate_rows))
    # plot_clean_df(df)
    return df_clean
    
# Everything is binarized now. It can be checked with this plot function    
def plot_clean_df(df):
    fig, axes = plt.subplots(6, 6, figsize=(6, 12))
    categories = df.columns
    for i in range(6):
        for j in range(6):
            sns.countplot(data=df, x=categories[j+i*6], ax=axes[i,j])
    plt.show()
    return None
    
def save_data(df, database_filename):
    engine = db.create_engine("sqlite:///"+database_filename)
    df.to_sql('Messages_Clean', engine, index=False, if_exists = "replace")
    pass  

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
    