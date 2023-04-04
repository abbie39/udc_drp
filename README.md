# udc_drp
Udacity Nanodegree - Disaster Response Pipeline project

## Project summary
This project uses tweets and text messages from disaster response situations. The aim is to help tackle the huge volumes of communications that are recieved directly following a disaster, to help disaster response organizations to filter out relevant messages. 
Different organisations are responsible for different elements of care and repair following a disaster - this is what determines the categories for the messages in the datasets. 
An ETL pipeline is used to clean and merge the messages and categories data.
A ML pipeline is used to build a supervised learning model, namely a RandomForestClassifier with multioutput capabilities. 
Finally a flask web app is responsible for the final user interface. It features the option to classify a message yourself to see which categories it fits into. Additionally it has two bar chart visualisations. One shows the volume of messages in the 'search and rescue' category vs. not in this category. The other shows the spread of how messages are received.

## How to run the python scripts and the web app
1. Download all files from the repo. Save them in the following way:
      a. Create a folder named udc_drp (you can call this whatever you like, I am giving it a name to make it easier to reference)
      b. Inside this folder create 3 subfolders: app, data, model
      c. Inside the app folder save the file: run.py
      d. Inside the data folder save the files: process_data.py, disaster_messages.csv, disaster_categories.csv, DisasterResponse.db
      e. Inside the model folder save the file: train_classifier.py
3. In your terminal navigate to the udc_drp folder.
4. Run <python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db>
6. Run <python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl>
7. Navigate to the app folder via the terminal (use <cd app>)
8. Run <python run.py>
9. Click preview button to view the web app


## Explanation of files
disaster_messages.csv : the csv file containing the messages data
disaster_categories.csv: the csv file containing categories data
DisasterResponse.db : SQL database used in the python scripts for saving clean data to
process_data.py : ETL pipeline script which loads, cleans, merges and saves the messages and categories data to a SQL database
train_classifier.py : trains a RandomForestClassifier model to categorise messages
run.py : Runs the flask web app

## Data sources
The data was generously provided by Appen (formerly FigureEight).
Tweets and texts were pre-labelled. 
