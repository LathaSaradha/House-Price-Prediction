'''
Author : Latha Saradha
Purpose : This file is a ML Models testing for all the different combination of features and create the CSV File
ML Models Used : Knn, Linear, SVM, XGBoost, RandomForest
'''


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import time

import math
import pathlib

sns.set(font_scale=0.5)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR



path= pathlib.Path().absolute()/"Data"/"Additional_Data"

class MLModels_with_std:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 2000)
    params = {'legend.fontsize': 10,
              'legend.handlelength': 1}
    plt.rcParams.update(params)



    def __init__(self):
        self.df_add_house_data_file = {}
        # self.df_ML_errors={}
        self.df_ML_errors = pd.DataFrame(columns=["Method", "R^2", "Adjusted R^2", "MAE", "MSE", "RMSE","HuberLoss",
             "logcosh", "Percent_Error" ,"ColsList"   ,"sigmoid % Error"     ])

    # Set the working directory
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


    #Method to load the combined house and external features data
    def load_combined_data(self, filename):
        print('Reading', filename)
        self.df_add_house_data_file = (pd.read_csv(filename, index_col=False))

        self.df_add_house_data_file = self.df_add_house_data_file[~self.df_add_house_data_file.isna()]

        print(self.df_add_house_data_file.head())
        print(self.df_add_house_data_file.shape)
        print(self.df_add_house_data_file.columns)

        print("Values with null")
        null_columns = self.df_add_house_data_file.columns[self.df_add_house_data_file.isnull().any()]
        cols = ['ZIP_OR_POSTAL_CODE', 'CITY', 'Num_of_Retail_stores_Zipcode']
        print(self.df_add_house_data_file[self.df_add_house_data_file["Num_of_Retail_stores_Zipcode"].isnull()][cols])



    # Method to calculate Linear Regression
    def LinearRegression1(self, X_train, X_test, Y_train, Y_test, list_of_columns,colslist):
        print('---------------------------------------------')
        print('LinearRegression1')
        # values converts it into a numpy array
        x_train = X_train[list_of_columns]
        x_test = X_test[list_of_columns]

        y_train = Y_train
        y_test = Y_test

        linear_regressor = LinearRegression(fit_intercept=True)  # create object for the class
        linear_regressor.fit(x_train, y_train)  # perform linear regression
        Y_pred = linear_regressor.predict(x_test)  # make predictions
        print('---------------------------------------------')
        '''
        print('Coeff :', linear_regressor.coef_)
        print('Intercept', linear_regressor.intercept_)
        print('LScore', linear_regressor.score(x_test, y_test))


        print('---------------------------------------------')
        print('Evaluation of Test Data')
        '''
        y_test_pred = linear_regressor.predict(x_test)
        # Model Evaluation
        self.FindErrors(x_test, y_test, y_test_pred, 'Linear Regressor',colslist)

    # Method to split the data set into two with dependent and independent variables data frame
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

    # Method to calculate the errors
    def FindErrors(self, x_value, y_value, y_value_pred, method,colslist):

        acc_linreg = metrics.r2_score(y_value, y_value_pred)
        adjusted_r2 = 1 -  ( (1 - (metrics.r2_score(y_value, y_value_pred)**2)) * (len(y_value) - 1) / (
                    len(y_value) - x_value.shape[1] - 1)  )
        mae = metrics.mean_absolute_error(y_value, y_value_pred)
        mse = metrics.mean_squared_error(y_value, y_value_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_value, y_value_pred))
        accuracy=round((acc_linreg * 100.0),2)

        delta=1.5

        huberLoss=self.huber(y_value,y_value_pred,delta)

        logcosh=self.logcosh(y_value,y_value_pred)


        percent = 15

        count_of_error_prediction = self.findpercentCount(y_value, y_value_pred, percent)
        sigmoid_error = self.findpercentCount_Sigmoid(y_value, y_value_pred, percent)

        percent_error = count_of_error_prediction / y_value_pred.shape[0]
        percent__sigmoid_error = sigmoid_error / y_value_pred.shape[0]

        print('more than', percent, ' percent error:', percent_error, y_value_pred.shape[0])

        # f1score=metrics.f1_score(y_value,y_value_pred)
        print('R^2:', acc_linreg)
        print('Adjusted R^2:', adjusted_r2)
        print('MAE:', mae)
        print('MSE:', mse)
        print('RMSE:', rmse)
        #print('huberLoss:  ' ,huberLoss)
        #print('logcosh: ',logcosh)
        print('Accuracy:  %.2f%%' % accuracy)
        print('Sigmoid Error',percent__sigmoid_error)

        self.df_ML_errors = self.df_ML_errors.append(
            {'colslist':colslist,'Method': method, 'R^2': acc_linreg, 'Adjusted R^2': adjusted_r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse,'Accuracy':accuracy,'HuberLoss':huberLoss,
             'logcosh':logcosh, 'Percent_Error':percent_error,
             'sigmoid % Error':percent__sigmoid_error

             },
            ignore_index=True)

    # Method to calculate huberloss
    def huber(self,trueValue, pred, delta):
        loss = np.where(np.abs(trueValue - pred) < delta, 0.5 * ((trueValue - pred) ** 2),
                        delta * np.abs(trueValue - pred) - 0.5 * (delta ** 2))
        return np.sum(loss)

    #  Method to calculate log cosh loss
    def logcosh(self,true_value, pred):
        loss = np.log(np.cosh(pred - true_value))
        return np.sum(loss)

    #  Method to calculate the percent error
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

    #  Method to calculate percent count using sigmoid function
    def findpercentCount_Sigmoid(self, true_value, pred, percent):
        #print("Finding percent count")

        true_value = true_value.to_numpy()
        print('converting to inverse standardisation')
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

    #  Method to create the histogram residuals
    def histogram_Residuals(self, axis):
        plt.hist(axis)
        plt.title("Histogram of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")

    #  Method to calculate Random Regression Model
    def RandomRegressor(self, X_train, X_test, Y_train, Y_test,colslist):
        print('---------------------------------------------')
        print('RandomRegressor')
        reg = RandomForestRegressor()

        # Train the model using the training sets  
        reg.fit(X_train, Y_train)

        #print('Evaluation of Test Data')
        y_test_pred = reg.predict(X_test)


        self.FindErrors(X_test, Y_test, y_test_pred, 'Random Regressor',colslist)



    #  Method to calculate XGBoost Regression
    def XGBoost_Regressor(self, X_train, X_test, Y_train, Y_test,colslist):
        print('---------------------------------------------')
        print('XGBoost_Regressor')
        reg = XGBRegressor(objective='reg:squarederror',
                           )

        # Train the model using the training sets
        reg.fit(X_train, Y_train)


        print('---------------------------------------------')
        #print('Evaluation of Test Data')

        y_test_pred = reg.predict(X_test)
        self.FindErrors(X_test, Y_test, y_test_pred, 'XGBoost_Regressor',colslist)
        print('---------------------------------------------')


    #  Method to calculate KNN model
    def KNN(self, X_train, X_test, Y_train, Y_test,colslist):
        print('---------------------------------------------')
        print('knn')
        knn = KNeighborsRegressor(n_neighbors=6)

        # Train the model using the training sets
        knn.fit(X_train, Y_train)

        print('---------------------------------------------')
        #print('Evaluation of Test Data')

        y_test_pred = knn.predict(X_test)
        self.FindErrors(X_test, Y_test, y_test_pred, 'knn',colslist)
        print('---------------------------------------------')

    #  Method to calculate data standardisation
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

        print('Eigen value of Standardized Dataset')
        self.calculationofEigenvalues(corr__std_matrix, self.df_add_house_data_file)

        self.df_add_house_data_file['PRICE'] = Y

    #  Method to calculate standardization
    def FindStandardizedDataset(self, numericvals):

        standardized_X = preprocessing.scale(numericvals)

        print(numericvals.columns.values)

        self.Mean = numericvals['PRICE'].mean()
        print("mean is ", self.Mean)
        self.std = np.std(numericvals['PRICE'])
        print("STD is ", self.std)

        print(standardized_X)

        std = pd.DataFrame(data=standardized_X, columns=numericvals.columns.values)
        print(std)
        return std

    #  Method to calculate Eigen values
    def calculationofEigenvalues(self, corr__std_matrix, standardized_dataset):
        eig_vals_std, eig_vecs_std = np.linalg.eig(corr__std_matrix)

        median = self.MedianOfEigenValues(eig_vals_std)
        new_array = np.vstack([standardized_dataset.columns.values, eig_vals_std])
        print(new_array)
        self.findValuesgreaterThanMedian(median, new_array)

    #  Method to calculate features greater than median for eigen values
    def findValuesgreaterThanMedian(self, median, new_array):

        print('--------------------------------------------------------------------')

        print("Features with eigen values > median")
        for i in range(0, len(new_array[0])):
            if (new_array[1][i] >= median):
                print(new_array[0][i])

    #  Method to calculate the median of eigen values
    def MedianOfEigenValues(self, eig_vals_std):
        print("Median of eigen values")
        median = np.median(eig_vals_std)
        print(median)
        return median

    #  Method to calculate correlation and plot it
    def correlation_plot_combined_file(self):
        print(self.df_add_house_data_file.columns)
        corr = self.df_add_house_data_file.corr()
        corr = corr.round(2)
        print(corr)
        print(type(corr))

        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )

        return corr

    #  Method to print the Errors
    def print_ML_errors(self):
        print(self.df_ML_errors)
        print(self.df_ML_errors.shape)

    #  Method to plot the ML errors
    def plot_ML_errors(self):
        print('Plotting errors')
        temp=self.df_ML_errors[["R^2", "Adjusted R^2", "MAE", "MSE", "RMSE","Percent_Error"]].copy()
        temp['RMSE']=1-temp['RMSE']
        temp['MAE'] = 1 - temp['MAE']
        temp['MSE'] = 1 - temp['MSE']
        temp['Percent_Error']=1-temp['Percent_Error']

        tempX=self.df_ML_errors['Method']
        plt.plot(tempX,temp)
        plt.xlabel('Method')
        plt.ylabel('Values')
        plt.title('ML errors')
        plt.legend()

    #  Method to calculate SVM model
    def SVM(self, X_train, X_test, Y_train, Y_test,colslist):
        print('---------------------------------------------')
        print('SVM Model')
        svm = LinearSVR(max_iter=10000)

        # Train the model using the training sets
        svm.fit(X_train, Y_train)
        print('---------------------------------------------')
        #print('Evaluation of Test Data')

        y_test_pred = svm.predict(X_test)
        self.FindErrors(X_test, Y_test, y_test_pred, 'svm',colslist)
        print('---------------------------------------------')

    #  Method to calculate the different loops for different sets
    def findingloops(self,X,Y):
        print("inside finding loops")
        rows = X.shape[0]
        cols = X.shape[1]
        listcolumns = X.columns
        print("columns are ")
        print(listcolumns)
        cols1 = []
        cols2 = []
        for i in range(0, cols):
            cols1.append(i)
            print("i is ",i)
            print(cols1, " and", cols2)
            tempX = self.df_add_house_data_file.iloc[0:rows, cols1 + cols2]
            self.callingMLModels(tempX, Y, listcolumns[cols1 + cols2])
            print("columns are ", listcolumns[cols1 + cols2])
            for j in range(i + 1, cols):
                cols2.append(j)
                print(cols1, " and", cols2)
                tempX=self.df_add_house_data_file.iloc[0:rows,cols1+cols2]
                self.callingMLModels(tempX, Y,listcolumns[cols1+cols2])
                print("columns are ", listcolumns[cols1 + cols2])

            cols1 = []
            cols2 = []
        print("Finishing Loop")
        self.print_ML_errors()
        self.plot_ML_errors()
        self.create_ML_Error_csv()

    #  Method to call ML models
    def callingMLModels(self,X, Y,colslist):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
        list_of_columns = X.columns
        self.LinearRegression1(X_train, X_test, Y_train, Y_test,list_of_columns,colslist)
        self.XGBoost_Regressor(X_train, X_test, Y_train, Y_test,colslist)
        self.RandomRegressor(X_train, X_test, Y_train, Y_test,colslist)
        self.KNN(X_train, X_test, Y_train, Y_test,colslist)
        self.SVM(X_train, X_test, Y_train, Y_test,colslist)
        #self.print_ML_errors()

    #  Method to create a CSV file with results
    def create_ML_Error_csv(self):
        print("copying the dataframe to a new csv file")
        # print(list(self.df_combined_file['ZIP_OR_POSTAL_CODE'].unique()))

        self.df_ML_errors.to_csv("ML Errors.csv", index=False)

def main():
    print("inside Main")
    obj = MLModels_with_std()
    obj.set_dir(path)

    obj.load_combined_data("AdditionalDataAndHouseData.csv")

    # Finding Standardisation
    obj.standardise_data()

    X, Y = obj.removePrice()
    start_time = time.time()

    obj.findingloops(X,Y)
    print("--- %s seconds for running all loops ---" % (time.time() - start_time))

    obj.plot_ML_errors()


if __name__ == '__main__':
    main()
