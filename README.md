# Disaster Response Pipeline Project

## Project Overview
The goal of this project is to classify disaster messages into categories. It utilizes the dataset provided by Figure Eight to build a classifier for a web API.  The user inputs the messages through the web API, and results are presented in several categories.
## Project Components:
1. ETL pipeline: is implemented in `process_data.py` file
- It loads the two datasets, namely: messages and categories
- Merge the datasets and clean it
- Store the cleaned dataset into SQLite database

2. ML pipeline: is implemented in `train_classifier.py` file
- Load the clean data from the SQLite database 
- Split the data into train-test sets
- Build a text processing and ML model
- Train and evaluate the model
- Dump the trained model 

3. Web app using Flask: is implemented in `run.py`
- It provides a platform for user to input a disaster message
and view the categories as output

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
