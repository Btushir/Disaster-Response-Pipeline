import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Load the clean and merged dataset of messages and categories from sqlite database
    Params: Database file path
    Return: features dataframe, target classes dataframe and category names (or target class names)
    """
    engine = create_engine('sqlite:///' + database_filepath).connect()
    df = pd.read_sql_table('disaster_response_clean_data_db', engine)
    # input features
    X = df.iloc[:, 1]
    # target classes
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the raw messages.
    Params: text (messages)
    Return: clean tokenize text
    """
    # convert all alphabet to lower case
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize the text
    text = word_tokenize(text)
    # remove stop words
    text = [w for w in text if w not in stopwords.words("english")]
    # perform lemmatization
    text = [WordNetLemmatizer().lemmatize(w, pos='v') for w in text]
    return text


def build_model():
    """
    Define a pipeline that includes text processing steps and classifier (random forest).
    Return: pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model and print the classification report for each
    Params: trained model, test samples and category names
    Return: None
    """
    y_pred = model.predict(X_test)
    # classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))
    # accuracy score
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy score is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    """
    save the trained model as pickel file
    Params: model: classifier to be saved and model_filepath
    Return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
