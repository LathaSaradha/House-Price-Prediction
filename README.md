# House-Price-Prediction


# Introduction 
In this project, an analysis of real-life housing price prediction is conducted based on the data collected from New York City. Along with the data collected from 'Redfin' regarding the price and features of a house, external data sets are incorporated, which includes information about the surrounding area. Combining all these features, data visualization is done to the cleaned and pre-processed data to identify the trends and patterns. Important features are studied, identified, and selected to attain dimensionality reduction. Following this, different machine learning models are applied to the datasets. For each model, the performance metrics are analyzed. It is concluded that the "Random Regression Model" predicts the house price with the most accuracy and the fewest errors among the selected models.


# Installation
Follow the steps below for the installation steps.

1. Install Python 3.7 from below website.

https://www.python.org/downloads/release/python-377/


2. Clone the code from 
https://github.com/LathaSaradha/House-Price-Prediction.git

3. 


# File Structure

1. loadData.py - This file is used to load the house information datasets which are available in csv format
2. cleanData.py -This file is used to load the CSV files which contain the data for the house
 and perform data pre-processing for the house features.
3. additionalDataFile.py - This file is used to load the CSV files containing additional information around the city and locality of house and cleaning datasets.
4. ###### MainFile.py -  
This file is a single point of file to call all the loading and
cleaning of data files and combine the house and external features.
5. ###### MLModelsTest.py - 
This file is a ML Models testing for all the different combination of features and create the CSV File
ML Models Used : Knn, Linear, SVM, XGBoost, RandomForest

6. 