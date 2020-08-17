# House-Price-Prediction


# Introduction 
In this project, an analysis of real-life housing price prediction is conducted based on the data collected from New York City. Along with the data collected from 'Redfin' regarding the price and features of a house, external data sets are incorporated, which includes information about the surrounding area. Combining all these features, data visualization is done to the cleaned and pre-processed data to identify the trends and patterns. Important features are studied, identified, and selected to attain dimensionality reduction. Following this, different machine learning models are applied to the datasets. For each model, the performance metrics are analyzed. It is concluded that the "Random Regression Model" predicts the house price with the most accuracy and the fewest errors among the selected models.


# Installation
Follow the steps below for the installation steps.

1. Install Python 3.7 from below website. https://www.python.org/downloads/release/python-377/

2. Clone the code from 
https://github.com/LathaSaradha/House-Price-Prediction.git

3. Install your choice of IDE
3. install the following libraries with versions

matplotlib - 3.2.1
numpy - 1.18.4
pandas - 1.0.4
scikit-learn - 0.23.1
seaborn - 0.10.1
statsmodel - 0.11.1



# File Structure

1. loadData.py - This file is used to load the house information datasets which are available in csv format
2. cleanData.py -This file is used to load the CSV files which contain the data for the house
 and perform data pre-processing for the house features.
3. additionalDataFile.py - This file is used to load the CSV files containing additional information around the city and locality of house and cleaning datasets.
4. ###### MainFile.py -  
This file is a single point of file to call all the loading and
cleaning of data files and combine the house and external features.
5. ###### MLModels_with_std.py - 
This file is a ML Models testing for all the different combination of features and create the CSV File
ML Models Used : Knn, Linear, SVM, XGBoost, RandomForest using standardization
Input - AdditionalDataAndHouseData.csv
Output -ML Errors.csv
6.  ###### MLModels_without_std.py - 
This file is a ML Models testing for all the different combination of features and create the CSV File
ML Models Used : Knn, Linear, SVM, XGBoost, RandomForest without standardization
Input - AdditionalDataAndHouseData.csv
Output -ML_Errors_without_standardisation.csv

6.  ###### LR_with_diff_alpha.py - 
This file is to find the ML performance for Linear Regression with different alpha values.
Input - AdditionalDataAndHouseData.csv
Output -ML_Errors_Linear_alpha.csv

7. ###### Graphs.py - 
This file is used to create Graphs or visual representation of the ML model performance and analysis of data

Run the files in following order

1. MainFile.py
2. MLModels_with_std.py
3. MLModels_without_std.py
4. LR_with_diff_alpha.py
5. Graphs.py

# Report

Available in Reports-> Project Report.docx


