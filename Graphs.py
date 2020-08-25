'''
Author : Latha Saradha
Purpose : This file is used to create Graphs or visual representation of the ML model performance and analysis of data
'''

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

import math


sns.set(font_scale=2.0)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
import pathlib


path= pathlib.Path().absolute()/"Data"/"Additional_Data"


class Graphs:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 2000)
    params = {'legend.fontsize': 12,
              'legend.handlelength': 1}
    plt.rcParams.update(params)

    def __init__(self):
        self.df_add_house_data_file = {}
        # self.df_ML_errors={}

        self.df_ML_errors = pd.DataFrame(columns=["Method", "R^2", "Adjusted R^2", "MAE", "MSE", "RMSE",
                                            "Percent_Error", "ColsList", "sigmoid % Error"])

        self.df_columnsList=pd.DataFrame(columns=["collist","setNum"])

    # Method to set the working directory
    def set_dir(self, path):
        try:
            os.chdir(path)
            print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")

        print(os.getcwd())
        if os.path.exists(path):
            # Change the current working Directory
            os.chdir(path)
        else:
            print("Can't change the Current Working Directory")

    # Method to load the combined data
    def load_combined_data(self, filename):
        print('Reading', filename)
        self.df_add_house_data_file = (pd.read_csv( filename, index_col=False))

        self.df_add_house_data_file = self.df_add_house_data_file[~self.df_add_house_data_file.isna()]

        print(self.df_add_house_data_file.head())
        print(self.df_add_house_data_file.shape)
        print(self.df_add_house_data_file.columns)

        print("Values with null")
        null_columns = self.df_add_house_data_file.columns[self.df_add_house_data_file.isnull().any()]
        cols = ['ZIP_OR_POSTAL_CODE', 'CITY', 'Num_of_Retail_stores_Zipcode']
        print(self.df_add_house_data_file[self.df_add_house_data_file["Num_of_Retail_stores_Zipcode"].isnull()][cols])

    # Method to create the graphs histogram of price, Distribution of house prices based on
    # number of beds and baths and correlation of features.
    def creategraphs(self):
        # Histogram of Beds and Baths and 1p PRICE
        self.df_add_house_data_file['PRICE'].plot.hist()
        plt.title('Histogram of Price')
        plt.xlabel('Price of House (Value in Million Dollars)')
        plt.ylabel('Number of houses')


        plt.show()


        ax=sns.boxplot(x=self.df_add_house_data_file['BATHS'], y=self.df_add_house_data_file['PRICE'], data=pd.melt(self.df_add_house_data_file))

        plt.title('Distribution of house prices based on number of baths')
        plt.xlabel('Number of Baths')
        plt.ylabel('Price of house (Value in Million Dollars)')
        plt.xticks(fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.show()

        ax = sns.boxplot(x=self.df_add_house_data_file['BEDS'], y=self.df_add_house_data_file['PRICE'],    data=pd.melt(self.df_add_house_data_file))
        plt.title('Distribution of house prices based on number of beds')
        plt.xlabel('Number of Beds')
        plt.ylabel('Price of house (Value in Million Dollars)')
        #plt.xticks(fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.show()

        print('Default')
        corr = self.df_add_house_data_file.corr()
        corr = corr.round(2)
        print(corr)
        print(type(corr))

        print('Printing correlation ')
        self.correlation_plot_combined_file()

    # Method to calculate the plot for correlation of features
    def correlation_plot_combined_file(self):
        print(self.df_add_house_data_file.columns)
        corr = self.df_add_house_data_file.corr()
        corr = corr.round(2)
        print(corr)
        print(type(corr))
        mask = np.triu(np.ones_like(corr, dtype=np.bool))


        sns.set(font_scale=0.5)
        ax=sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(220, 10, n=200),
            square=True,
            linewidths=.5

        )
        #plt.figure(figsize=(10, 5))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        ax.set_title('Correlation of features',fontsize=13)

        plt.show()
        return corr



    # Method to split the dataset to dependent and independent variable features
    def removePrice(self):
        X = self.df_add_house_data_file.drop(['PRICE'], axis=1)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        numericvals = X.select_dtypes(include=numerics)
        print('--------------------------------------------------------------------')
        print('Printing numeric vals')
        print(numericvals)
        print(numericvals.head)

        Y = self.df_add_house_data_file['PRICE']
        print(Y.head)

        return numericvals, Y

    #Method to calculate Linear Regression Model
    def LinearRegression1(self, X_train, X_test, Y_train, Y_test, list_of_columns,colslist,method):
        print('---------------------------------------------')
        print('LinearRegression1')
        # values converts it into a numpy array
        x_train = X_train[list_of_columns]
        x_test = X_test[list_of_columns]

        y_train = Y_train
        y_test = Y_test
        print(x_train.columns)

        linear_regressor = LinearRegression(fit_intercept=True)  # create object for the class
        linear_regressor.fit(x_train, y_train)  # perform linear regression
        Y_pred = linear_regressor.predict(x_test)  # make predictions
        print('---------------------------------------------')
        print('Coeff :', linear_regressor.coef_)
        print('Intercept', linear_regressor.intercept_)
        print('LScore', linear_regressor.score(x_test, y_test))


        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = linear_regressor.predict(x_test)
        # Model Evaluation
        self.FindErrors(x_test, y_test, y_test_pred, method,colslist)

    # Method to standardize data
    def standardise_data(self):

        numeric = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        numerical = self.df_add_house_data_file.select_dtypes(include=numeric)

        print("Numeric columns are.....")

        numerical = numerical.reset_index(drop=True)
        print(numerical)

        # Standardising the data
        print('--------------------------------------------------------------------')

        self.df_add_house_data_file = self.FindStandardizedDataset(numerical)
        print('--------------------------------------------------------------------')
        print('Correlation Matrix of Standardized Dataset')

        Y = self.df_add_house_data_file['PRICE']

        to_drop2 = ['PRICE']

        print(self.df_add_house_data_file.columns)

        self.df_add_house_data_file = self.df_add_house_data_file.drop(['PRICE'], axis=1)

        print(self.df_add_house_data_file.columns)

        corr__std_matrix = self.correlation_plot_combined_file()


        self.df_add_house_data_file['PRICE'] = Y

    # Method to find the standardize data set
    def FindStandardizedDataset(self, numericvals):

        standardized_X = preprocessing.scale(numericvals)
        # print(standardized_X)
        # print(standardized_X.dtype)
        print(numericvals.columns.values)

        self.Mean = numericvals['PRICE'].mean()
        print("mean is ", self.Mean)
        self.std = np.std(numericvals['PRICE'])
        print("STD is ", self.std)

        print(standardized_X)

        std = pd.DataFrame(data=standardized_X, columns=numericvals.columns.values)
        print(std)
        return std


    #Method to create Linear graph
    def create_Linear_graph(self):
        #self.standardise_data()
        X, Y = self.removePrice()
        rows = X.shape[0]
        self.df_ML_errors = self.df_ML_errors.iloc[0:0]

        print('Priniting for checking the error dataframe')

        print(self.df_ML_errors)

        for x in range(self.df_columnsList.shape[0]):
            colslist = self.df_columnsList['collist'][x]
            setNum = self.df_columnsList['setNum'][x]
            print(colslist)
            print(setNum)
            self.call_LinearRegression(Y, colslist, setNum)

        self.plot_ML_errors('Linear Regression Sets')
        self.print_ML_errors()

    # Method to call Linear Regression
    def call_LinearRegression(self, Y, colslist,set):
        tempX = self.df_add_house_data_file[colslist]
        X_train, X_test, Y_train, Y_test = train_test_split(tempX, Y, test_size=0.2, random_state=4)
        list_of_columns = X_train.columns
        self.LinearRegression1(X_train, X_test, Y_train, Y_test, list_of_columns, colslist, set)

    # Method to call Random Forest Regression
    def call_RandomForest_Regression(self, Y, colslist,set):
        tempX = self.df_add_house_data_file[colslist]
        X_train, X_test, Y_train, Y_test = train_test_split(tempX, Y, test_size=0.2, random_state=4)
        list_of_columns = X_train.columns
        self.RandomRegressor(X_train, X_test, Y_train, Y_test, colslist, set)

    # Method to call SVM regression
    def call_SVM_Regression(self, Y, colslist,set):
        tempX = self.df_add_house_data_file[colslist]
        X_train, X_test, Y_train, Y_test = train_test_split(tempX, Y, test_size=0.2, random_state=4)
        list_of_columns = X_train.columns
        self.SVM(X_train, X_test, Y_train, Y_test, colslist, set)

    # Method to plot ML errors
    def plot_ML_errors(self,xlabel):
        print('Plotting errors')

        tempDF=self.df_ML_errors

        temp=tempDF[["R^2", "Adjusted R^2", "MAE", "MSE", "RMSE","Percent_Error","sigmoid % Error","Accuracy"]].copy()
        temp['R^2']=1-temp['R^2']
        temp['Adjusted R^2'] = 1 - temp['Adjusted R^2']
        temp['Accuracy'] = 1- (temp['Accuracy']/100)

        temp = temp.rename(columns={"R^2": "1- R^2"})
        temp = temp.rename(columns={"Adjusted R^2": "1- Adjusted R^2"})
        temp = temp.rename(columns={"Accuracy": "1- Accuracy"})
        tempX = tempDF['Method']
        plt.plot(tempX, temp)
        plt.xlabel(xlabel,fontsize=14)
        plt.ylabel('Performance Metrics',fontsize=14)
        plt.title('Performance Evaluation',fontsize=14)
        params = {'legend.fontsize': 14,
                  'legend.handlelength': 1}
        plt.rcParams.update(params)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        leg = plt.legend()
        plt.legend(temp.columns)
        # temp.plot(x='Method',figsize=(20,20))

        plt.show()

    # Method to print ML errors
    def print_ML_errors(self):
        print(self.df_ML_errors)
        print(self.df_ML_errors.shape)

    # Method to find ML errors
    def FindErrors(self, x_value, y_value, y_value_pred, method,colslist):

        acc_linreg = metrics.r2_score(y_value, y_value_pred)
        adjusted_r2 = 1 -  ( (1 - (metrics.r2_score(y_value, y_value_pred)**2)) * (len(y_value) - 1) / (
                    len(y_value) - x_value.shape[1] - 1)  )
        mae = metrics.mean_absolute_error(y_value, y_value_pred)
        mse = metrics.mean_squared_error(y_value, y_value_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_value, y_value_pred))
        accuracy=round((acc_linreg * 100.0),2)


        percent = 15

        count_of_error_prediction = self.findpercentCount(y_value, y_value_pred, percent)
        sigmoid_error = self.findpercentCount_Sigmoid(y_value, y_value_pred, percent)

        percent_error = count_of_error_prediction / y_value_pred.shape[0]
        percent__sigmoid_error = sigmoid_error / y_value_pred.shape[0]

        # commented to reduce prints

        print('more than', percent, ' percent error:', percent_error, y_value_pred.shape[0])

        # f1score=metrics.f1_score(y_value,y_value_pred)
        print('R^2:', acc_linreg)
        print('Adjusted R^2:', adjusted_r2)
        print('MAE:', mae)
        print('MSE:', mse)
        print('RMSE:', rmse)

        print('Accuracy:  %.2f%%' % accuracy)
        print('Sigmoid Error',percent__sigmoid_error)


        self.df_ML_errors = self.df_ML_errors.append(
            {'colslist':colslist,'Method': method, 'R^2': acc_linreg, 'Adjusted R^2': adjusted_r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse,
             'Accuracy':accuracy,
              'Percent_Error':percent_error,
             'sigmoid % Error':percent__sigmoid_error

             },
            ignore_index=True)

    # Method to find percentage of error
    def findpercentCount(self, true_value, pred, percent):
        print("Finding percent count")

        true_value = true_value.to_numpy()
        #print('converting to inverse standardisation')
        count = 0
        for row in range(pred.shape[0]):
            #print(row)
            y_truevalue = (true_value[row] * self.std) + self.Mean
            y_predvalue = (pred[row] * self.std) + self.Mean
            # print(y_truevalue," ",y_predvalue,"  from ",true_value[row],"  ",pred[row])
            percentvalue = (y_truevalue * percent) / 100

            diff = abs(y_truevalue - y_predvalue)
            if (diff > percentvalue):
                count += 1
            # print(diff)
        return count

    # Method to percentage of error using sigmoid function
    def findpercentCount_Sigmoid(self, true_value, pred, percent):
        print("Finding sigmoid percent count")

        true_value = true_value.to_numpy()
        #print('converting to inverse standardisation')
        count = 0
        for row in range(pred.shape[0]):
            # print(row)
            y_truevalue = (true_value[row] * self.std) + self.Mean
            y_predvalue = (pred[row] * self.std) + self.Mean
            # print(y_truevalue," ",y_predvalue,"  from ",true_value[row],"  ",pred[row])
            ytemp=y_truevalue/1000000
            percent=0.3*(math.exp(-ytemp))
            #print(y_truevalue," ",percent*100)
            percent=percent*100
            percentvalue = (y_truevalue * percent) / 100

            diff = abs(y_truevalue - y_predvalue)
            if (diff > percentvalue):
                count += 1
            # print(diff)
        return count

    # Method to calculate Random Regression
    def RandomRegressor(self, X_train, X_test, Y_train, Y_test, colslist,method):
        print('---------------------------------------------')
        print('RandomRegressor')
        reg = RandomForestRegressor()

        # Train the model using the training sets
        reg.fit(X_train, Y_train)

        print('Evaluation of Test Data')
        y_test_pred = reg.predict(X_test)

        self.FindErrors(X_test, Y_test, y_test_pred, method, colslist)

    # Method to create Random Forest Graph
    def create_Random_forest_graph(self):
        #self.standardise_data()
        X, Y = self.removePrice()
        rows = X.shape[0]


        self.df_ML_errors=self.df_ML_errors.iloc[0:0]

        print('Priniting for checking the error dataframe')

        print(self.df_ML_errors)


        for x in range(self.df_columnsList.shape[0]):
            colslist = self.df_columnsList['collist'][x]
            setNum =self.df_columnsList['setNum'][x]
            print(colslist)
            print(setNum)
            self.call_RandomForest_Regression(Y,colslist,setNum)


        self.plot_ML_errors('Random Regression Sets')
        self.print_ML_errors()

    # Method to create SVM graph
    def create_SVM_graph(self):
        # self.standardise_data()
        X, Y = self.removePrice()
        rows = X.shape[0]

        self.df_ML_errors = self.df_ML_errors.iloc[0:0]

        print('Priniting for checking the error dataframe')

        print(self.df_ML_errors)

        for x in range(self.df_columnsList.shape[0]):
            colslist = self.df_columnsList['collist'][x]
            setNum = self.df_columnsList['setNum'][x]
            print(colslist)
            print(setNum)
            self.call_SVM_Regression(Y, colslist, setNum)

        self.plot_ML_errors('SVM Regression Sets')
        self.print_ML_errors()

    # Method to calculate SVM ML model
    def SVM(self, X_train, X_test, Y_train, Y_test, colslist,method):
        print('---------------------------------------------')
        print('SVM Model')
        svm = LinearSVR(max_iter=10000)

        # Train the model using the training sets
        svm.fit(X_train, Y_train)


        print('---------------------------------------------')
        print('Evaluation of Test Data')

        y_test_pred = svm.predict(X_test)
        self.FindErrors(X_test, Y_test, y_test_pred, method, colslist)
        print('---------------------------------------------')

    # Method to calculate XG Boost Regression
    def create_XGBoost_graph(self):

        X, Y = self.removePrice()
        rows = X.shape[0]

        self.df_ML_errors = self.df_ML_errors.iloc[0:0]

        print('Printing for checking the error dataframe')

        print(self.df_ML_errors)

        for x in range(self.df_columnsList.shape[0]):
            colslist = self.df_columnsList['collist'][x]
            setNum = self.df_columnsList['setNum'][x]
            print(colslist)
            print(setNum)
            self.call_XGBoost_Regression(Y, colslist, setNum)

        self.plot_ML_errors('XGBoost Regression Sets')
        self.print_ML_errors()

    # Method to call XG Boost Regression
    def call_XGBoost_Regression(self, Y, colslist,set):
        tempX = self.df_add_house_data_file[colslist]
        X_train, X_test, Y_train, Y_test = train_test_split(tempX, Y, test_size=0.2, random_state=4)
        list_of_columns = X_train.columns

        self.XGBoost_Regressor(X_train, X_test, Y_train, Y_test, colslist, set)

    # Method to calculate KNN model
    def KNN(self, X_train, X_test, Y_train, Y_test, colslist,set):
        print('---------------------------------------------')
        print('knn')
        knn = KNeighborsRegressor(n_neighbors=6,weights='distance')

        # Train the model using the training sets
        knn.fit(X_train, Y_train)

        print('---------------------------------------------')
        print('Evaluation of Test Data')

        y_test_pred = knn.predict(X_test)
        self.FindErrors(X_test, Y_test, y_test_pred, set, colslist)
        print('---------------------------------------------')

    # Method to call KNN
    def call_KNN_Regression(self, Y, colslist,set):
        tempX = self.df_add_house_data_file[colslist]
        X_train, X_test, Y_train, Y_test = train_test_split(tempX, Y, test_size=0.2, random_state=4)
        list_of_columns = X_train.columns
        self.KNN(X_train, X_test, Y_train, Y_test, colslist, set)

    # Method to create KNN graph
    def create_KNN_graph(self):
        # self.standardise_data()
        X, Y = self.removePrice()
        rows = X.shape[0]

        self.df_ML_errors = self.df_ML_errors.iloc[0:0]

        print('Printing for checking the error dataframe')

        print(self.df_ML_errors)

        for x in range(self.df_columnsList.shape[0]):
            colslist = self.df_columnsList['collist'][x]
            setNum = self.df_columnsList['setNum'][x]
            print(colslist)
            print(setNum)
            self.call_KNN_Regression(Y, colslist, setNum)

        self.plot_ML_errors('Knn Regression Sets')
        self.print_ML_errors()

    # Method to calculate XG boost regression ML model

    def XGBoost_Regressor(self, X_train, X_test, Y_train, Y_test,colslist,method):
        print('---------------------------------------------')
        print('XGBoost_Regressor')
        reg = XGBRegressor(objective='reg:squarederror',
                           )

        # Train the model using the training sets
        reg.fit(X_train, Y_train)


        print('---------------------------------------------')
        print('Evaluation of Test Data')

        y_test_pred = reg.predict(X_test)
        self.FindErrors(X_test, Y_test, y_test_pred, method,colslist)

    # Method to define columns list to generate graphs
    def define_ColumnsList(self):

        temp={ 'collist':[ ['Population'],
                                            ['ZIP_OR_POSTAL_CODE'],
                                        ['Total_Num_ofHospitals'],
        ['Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount'],
         ['BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric'],
         ['Total_Num_of_Subways'],
        ['Total_Num_of_Subways', 'min_dist_station', 'Num_of_Retail_stores', 'min_dist_retail_store'],
        ['Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount', 'Level_C_SchoolCount',
                    'Level_D_SchoolCount',
                    'Level_F_SchoolCount', 'Total_Number_of_Schools', 'Num_Complaints_schools', 'Population',
                    'People/Sq_Mile',
                    'Total_Num_ofHospitals'],
         ['Level_F_SchoolCount', 'Total_Number_of_Schools', 'Num_Complaints_schools', 'Population',
                    'People/Sq_Mile',
                    'Total_Num_ofHospitals', 'Total_Num_of_Subways', 'min_dist_station', 'Num_of_Retail_stores',
                    'min_dist_retail_store'],
        ['SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric',
                    'Total_Num_ofComplaints',
                    'Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount', 'Level_C_SchoolCount'],
        ['SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric',
                    'Total_Num_ofComplaints',
                    'Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount', 'Level_C_SchoolCount',
                    'Level_D_SchoolCount', 'Level_F_SchoolCount', 'Total_Number_of_Schools',
                    'Num_Complaints_schools'],

        ['ZIP_OR_POSTAL_CODE', 'BEDS', 'BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE',
                    'AGE',
                    'CITY numeric', 'Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount'],

        ['BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric',
                    'Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount',
                    'Level_C_SchoolCount', 'Level_D_SchoolCount', 'Level_F_SchoolCount', 'Total_Number_of_Schools',
                    'Num_Complaints_schools', 'Population'],

       ['ZIP_OR_POSTAL_CODE', 'BEDS', 'BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE',
                    'AGE',
                    'CITY numeric', 'Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount',
                    'Level_B_SchoolCount',
                    'Level_C_SchoolCount', 'Level_D_SchoolCount', 'Level_F_SchoolCount', 'Total_Number_of_Schools',
                    'Num_Complaints_schools', 'Population', 'People/Sq_Mile', 'Total_Num_ofHospitals',
                    'Total_Num_of_Subways', 'min_dist_station', 'Num_of_Retail_stores', 'min_dist_retail_store'],
      ['BEDS', 'BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric',
                    'Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount',
                    'Level_C_SchoolCount', 'Level_D_SchoolCount', 'Level_F_SchoolCount', 'Total_Number_of_Schools',
                    'Num_Complaints_schools', 'Population', 'People/Sq_Mile', 'Total_Num_ofHospitals',
                    'Total_Num_of_Subways', 'min_dist_station', 'Num_of_Retail_stores', 'min_dist_retail_store'],
         ['BEDS', 'BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric',
                    'Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount',
                    'Level_C_SchoolCount', 'Level_D_SchoolCount', 'Level_F_SchoolCount', 'Total_Number_of_Schools',
                    'Num_Complaints_schools', 'Population', 'People/Sq_Mile', 'Total_Num_ofHospitals',
                    'Total_Num_of_Subways', 'min_dist_station', 'Num_of_Retail_stores', 'min_dist_retail_store',
                    'Num_of_Retail_stores_Zipcode'] ]

                              , 'setNum':['set 1','set 2','set 3','set 4','set 5','set 6',
                                           'set 7', 'set 8', 'set 9', 'set 10',
                                          'set 11', 'set 12',
                                          'set 13', 'set 14', 'set 15', 'set 16'

                                          ]
                              }
        self.df_columnsList =pd.DataFrame(temp)
        print('Defining the columns list')



def main():
    print("inside Main")
    obj = Graphs()
    obj.set_dir(path)

    obj.load_combined_data("AdditionalDataAndHouseData.csv")
    print("Calling graphs")


    obj.creategraphs()


    #Defining list of columns to be calculated
    obj.standardise_data()
    obj.define_ColumnsList()

    obj.create_Linear_graph()

    obj.create_Random_forest_graph()

    obj.create_SVM_graph()

    obj.create_XGBoost_graph()

    obj.create_KNN_graph()

if __name__ == '__main__':
    main()