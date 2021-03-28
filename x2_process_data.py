#Importing the libraries used
import pandas as pd
import numpy as np

defaultTest = pd.read_csv("test.csv")                # Importing the dataframe 'test.csv' from the appropriate folder
defaultTrain = pd.read_csv("train.csv")              # Importing the dataframe 'train.csv' from the appropriate folder
fulldf = pd.concat([defaultTest, defaultTrain])      # Merging  the 2 datasets, so that we can work as if we were given the undivided

#dataset in the first place.
fulldf.index.name = 'i'                              # Renaming the index column so that the new index will not have the same name as the old
fulldf = fulldf.reset_index()                        # Resetting the indexes for appearance's sake.
fulldf = fulldf.drop(['i', 'Unnamed: 0'], axis = 1)  # Removing the columns of indexes which we changed: the change was not necessary, only for the sake of appearance

# Data Cleaning
dupl = fulldf.groupby(['id']).size()>1
fulldf = fulldf.drop(['id'], axis = 1)               # Further unneded attribute

#Encoding values from string to numerical
fulldf['Gender'] = fulldf['Gender'].replace({"Male": 0, "Female": 1})
fulldf['satisfaction'] = fulldf['satisfaction'].replace({"neutral or dissatisfied": 0, "satisfied": 1})
fulldf['Type of Travel'] = fulldf['Type of Travel'].replace({"Personal Travel": 0, "Business travel": 1})
fulldf['Customer Type'] = fulldf['Customer Type'].replace({"disloyal Customer": 0, "Loyal Customer": 1})
fulldf['Class'] = fulldf['Class'].replace({"Eco": 0, "Eco Plus": 1, "Business": 2})

fulldf['Arrival Delay in Minutes'].fillna(fulldf['Departure Delay in Minutes'], inplace = True)

# Removing the departure delay in minutes - correlation is 0.96 - very high, no need for both as one follows mostly from the other
fulldf = fulldf.drop(['Departure Delay in Minutes'], axis = 1)

# Saving data
fulldf.to_csv('data_processed.csv', encoding='utf-8')

