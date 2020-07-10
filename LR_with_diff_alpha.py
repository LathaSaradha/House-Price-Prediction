import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import time
import statsmodels.api as sm
import pylab as py
import matplotlib.ticker as plticker
import math
from pandas import DataFrame

sns.set(font_scale=1.5)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.naive_bayes import GaussianNB

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/Additional Data/'


class MLModels:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 2000)
    params = {'legend.fontsize': 10,
              'legend.handlelength': 1}
    plt.rcParams.update(params)

    def __init__(self):
        self.df_add_house_data_file = {}
        # self.df_ML_errors={}
        self.df_ML_errors = pd.DataFrame(columns=["Method", "R^2", "Adjusted R^2", "MAE", "MSE", "RMSE",
                                                  "Percent_Error", "ColsList", "sigmoid % Error","alpha"])

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

    def load_combined_data(self, filename):
        print('Reading', filename)
        self.df_add_house_data_file = (pd.read_csv(path + filename, index_col=False))

        self.df_add_house_data_file = self.df_add_house_data_file[~self.df_add_house_data_file.isna()]

        print(self.df_add_house_data_file.head())
        print(self.df_add_house_data_file.shape)
        print(self.df_add_house_data_file.columns)

        print("Values with null")
        null_columns = self.df_add_house_data_file.columns[self.df_add_house_data_file.isnull().any()]
        cols = ['ZIP_OR_POSTAL_CODE', 'CITY', 'Num_of_Retail_stores_Zipcode']
        print(self.df_add_house_data_file[self.df_add_house_data_file["Num_of_Retail_stores_Zipcode"].isnull()][cols])

        # plotting QQ plot
        data_points = self.df_add_house_data_file[['PRICE']]

        sm.qqplot(data_points, line='45')
        # py.show()

    def LinearRegression1(self, X_train, X_test, Y_train, Y_test, list_of_columns, colslist):
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
        print('Coeff :', linear_regressor.coef_)
        print('Intercept', linear_regressor.intercept_)
        print('LScore', linear_regressor.score(x_test, y_test))

        # Commented to reject Train data evaluation
        '''

        print('Evaluation of Train Data')
        print('---------------------------------------------')
        y_pred_train = linear_regressor.predict(x_train)
        self.FindErrors(x_train, y_train, y_pred_train, 'Linear Regressor Train')
        '''

        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = linear_regressor.predict(x_test)
        # Model Evaluation
        alpha_val=0
        self.FindErrors(x_test, y_test, y_test_pred, 'Linear Regressor', colslist,alpha_val)




    def LinearRegression_Lasso(self, X_train, X_test, Y_train, Y_test, list_of_columns, colslist,alpha_val):
        print('---------------------------------------------')
        print('LinearRegression Lasso')
        # values converts it into a numpy array
        x_train = X_train[list_of_columns]
        x_test = X_test[list_of_columns]

        y_train = Y_train
        y_test = Y_test
        alpha=0.5
        linear_regressor = Lasso(fit_intercept=True,alpha=alpha_val)  # create object for the class
        linear_regressor.fit(x_train, y_train)  # perform linear regression
        Y_pred = linear_regressor.predict(x_test)  # make predictions
        print('---------------------------------------------')
        print('Coeff :', linear_regressor.coef_)
        print('Intercept', linear_regressor.intercept_)
        print('LScore', linear_regressor.score(x_test, y_test))

        # Commented to reject Train data evaluation
        '''

        print('Evaluation of Train Data')
        print('---------------------------------------------')
        y_pred_train = linear_regressor.predict(x_train)
        self.FindErrors(x_train, y_train, y_pred_train, 'Linear Regressor Train')
        '''

        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = linear_regressor.predict(x_test)
        # Model Evaluation
        self.FindErrors(x_test, y_test, y_test_pred, 'Linear Regressor Lasso', colslist,alpha_val)



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

    def FindErrors(self, x_value, y_value, y_value_pred, method, colslist,alpha_val):

        # plt.scatter(y_value, y_value_pred)
        # plt.xlabel("Prices")
        # plt.ylabel("Predicted prices")
        # title="Prices vs Predicted prices - "+method
        # plt.title(title)
        # plt.show()
        acc_linreg = metrics.r2_score(y_value, y_value_pred)
        adjusted_r2 = 1 - ((1 - (metrics.r2_score(y_value, y_value_pred) ** 2)) * (len(y_value) - 1) / (
                len(y_value) - x_value.shape[1] - 1))
        mae = metrics.mean_absolute_error(y_value, y_value_pred)
        mse = metrics.mean_squared_error(y_value, y_value_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_value, y_value_pred))
        accuracy = round((acc_linreg * 100.0), 2)

        delta = 1.5



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

        print('Accuracy:  %.2f%%' % accuracy)
        print('Sigmoid Error', percent__sigmoid_error)

        self.df_ML_errors = self.df_ML_errors.append(
            {'colslist': colslist, 'Method': method, 'R^2': acc_linreg, 'Adjusted R^2': adjusted_r2, 'MAE': mae,
             'MSE': mse, 'RMSE': rmse, 'Accuracy': accuracy,
             'Percent_Error': percent_error,
             'sigmoid % Error': percent__sigmoid_error,
             'alpha' :alpha_val

             },
            ignore_index=True)

    def huber(self, trueValue, pred, delta):
        loss = np.where(np.abs(trueValue - pred) < delta, 0.5 * ((trueValue - pred) ** 2),
                        delta * np.abs(trueValue - pred) - 0.5 * (delta ** 2))
        return np.sum(loss)

    # log cosh loss
    def logcosh(self, true_value, pred):
        loss = np.log(np.cosh(pred - true_value))
        return np.sum(loss)

    def findpercentCount(self, true_value, pred, percent):
        print("Finding percent count")

        true_value = true_value.to_numpy()
        print('converting to inverse standardisation')
        count = 0
        y_test=[]
        Y_pred=[]
        difference=[]
        for row in range(pred.shape[0]):
            # print(row)
            y_truevalue = (true_value[row] * self.std) + self.Mean
            y_test.append(y_truevalue)
            y_predvalue = (pred[row] * self.std) + self.Mean
            Y_pred.append(y_predvalue)

            # print(y_truevalue," ",y_predvalue,"  from ",true_value[row],"  ",pred[row])
            percentvalue = (y_truevalue * percent) / 100

            diff = abs(y_truevalue - y_predvalue)
            difference.append((diff))
            if (diff > percentvalue):
                count += 1
        print(type(y_test))
        #df_Y_Pred = pd.DataFrame(y_predvalue, columns=['Predicted'])
        #df_Y_True = pd.DataFrame(y_truevalue, columns=['True_Value'])

        # plt.figure(figsize=(20, 20))
        # plt.plot([1, 2, 3])
        # plt.subplot(221)
        # plt.scatter(y_test, Y_pred)
        # #plt.plot(df_Y_True, df_Y_Pred)
        # plt.xlabel("Prices")
        # plt.ylabel("Predicted prices")
        # plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, Y_pred, 1))(np.unique(y_test)),color='red')
        # plt.title("Prices vs Predicted prices -Linear")
        #
        #
        # plt.subplot(222)
        #
        # #plt.plot(df_Y_Pred, df_Y_True - df_Y_Pred)
        # plt.scatter(Y_pred, difference)
        # plt.title("Predicted vs residuals Linear")
        # plt.xlabel("Predicted")
        # plt.ylabel("Residuals")
        # #plt.show()
            # print(diff)
        return count

    def findpercentCount_Sigmoid(self, true_value, pred, percent):
        print("Finding percent count")

        true_value = true_value.to_numpy()
        print('converting to inverse standardisation')
        count = 0
        for row in range(pred.shape[0]):
            # print(row)
            y_truevalue = (true_value[row] * self.std) + self.Mean
            y_predvalue = (pred[row] * self.std) + self.Mean
            # print(y_truevalue," ",y_predvalue,"  from ",true_value[row],"  ",pred[row])
            ytemp = y_truevalue / 1000000
            percent = 0.3 * (math.exp(-ytemp))
            # print(y_truevalue," ",percent*100)
            percent = percent * 100
            percentvalue = (y_truevalue * percent) / 100

            diff = abs(y_truevalue - y_predvalue)
            if (diff > percentvalue):
                count += 1
            # print(diff)
        return count

    def histogram_Residuals(self, axis):
        plt.hist(axis)
        plt.title("Histogram of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")


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

    def calculationofEigenvalues(self, corr__std_matrix, standardized_dataset):
        eig_vals_std, eig_vecs_std = np.linalg.eig(corr__std_matrix)

        median = self.MedianOfEigenValues(eig_vals_std)
        new_array = np.vstack([standardized_dataset.columns.values, eig_vals_std])
        print(new_array)
        self.findValuesgreaterThanMedian(median, new_array)

    def findValuesgreaterThanMedian(self, median, new_array):
        print('--------------------------------------------------------------------')
        print("Features with eigen values > median")
        for i in range(0, len(new_array[0])):
            if (new_array[1][i] >= median):
                print(new_array[0][i])

    def MedianOfEigenValues(self, eig_vals_std):
        print("Median of eigen values")
        median = np.median(eig_vals_std)
        print(median)
        return median

    def correlation_plot_combined_file(self):
        print(self.df_add_house_data_file.columns)
        corr = self.df_add_house_data_file.corr()
        corr = corr.round(2)
        print(corr)
        print(type(corr))
        # commented to reduce prints
        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        # plt.show()
        return corr

    def print_ML_errors(self):
        print(self.df_ML_errors)
        print(self.df_ML_errors.shape)

    def plot_ML_errors(self):

        temp = self.df_ML_errors[["R^2", "Adjusted R^2", "MAE", "MSE", "RMSE", "Percent_Error","sigmoid % Error","alpha"]].copy()
        temp['R^2'] = 1 - temp['R^2']
        temp['Adjusted R^2'] = 1 - temp['Adjusted R^2']

        # locs, labels = plt.xticks(ticks=temp['Method'])
        tempX = self.df_ML_errors['Method']
        temp = temp.rename(columns={"R^2": "1-R^2"})
        temp=temp.rename(columns={"Adjusted R^2": "1-Adjusted R^2"})


        plt.plot(tempX, temp)
        plt.xlabel('Linear Regression Methods')
        plt.ylabel('Performance Metrics')
        plt.title('Performance Evaluation')
        plt.xticks(fontsize=12)
        leg=plt.legend()
        plt.legend(temp.columns)
        #temp.plot(x='Method',figsize=(20,20))

        plt.show()
        temp = self.df_ML_errors[
            ["R^2", "Adjusted R^2", "MAE", "MSE", "RMSE", "Percent_Error", "sigmoid % Error", "Accuracy","alpha",
             "Method"]].copy()
        temp['R^2'] = 1 - temp['R^2']
        temp['Adjusted R^2'] = 1 - temp['Adjusted R^2']
        temp['Accuracy'] = 1 - (temp['Accuracy'] / 100)

        temp = temp.rename(columns={"R^2": "1- R^2"})
        temp = temp.rename(columns={"Adjusted R^2": "1- Adjusted R^2"})
        temp = temp.rename(columns={"Accuracy": "1- Accuracy"})

        print('Plotting Ridge')

        #
        plt.figure(figsize=(20, 20))
        plt.suptitle('Performance Evaluation')
        plt.plot([1, 2, 3, 4])

        plt.subplot(221)


        type=['Linear Regressor Ridge']
        temp_Ridge=temp[temp.Method.isin(type)]
        print(temp_Ridge)
        temp_Ridge_df= temp_Ridge[["1- R^2", "1- Adjusted R^2", "MAE", "MSE", "RMSE", "Percent_Error","sigmoid % Error","1- Accuracy"]].copy()
        temp_Ridge['alpha'] = temp_Ridge['alpha'].astype(str)

        tempX_Ridge=temp_Ridge['alpha']
        ax=plt.plot(tempX_Ridge, temp_Ridge_df)
        plt.xlabel('Linear Regression Ridge',fontsize=12)
        plt.ylabel('Performance Metrics',fontsize=12)
        #plt.title('Performance Evaluation')
        label=temp_Ridge['alpha']
        plt.xticks(tempX_Ridge,labels=label,fontsize=12)
        plt.yticks(fontsize=12)

        #self.tick_space(0.50,'major','x')
        #plt.legend(temp_Ridge_df.columns)
        #plt.show()

        plt.subplot(222)

        print('plotting  lasso')
        type = ['Linear Regressor Lasso']

        temp_Lasso = temp[temp.Method.isin(type)]

        cols = [0,0.5,0.75,1.0,5,10,100,500,1000,10000]
        temp_Lasso = temp_Lasso[temp_Lasso.alpha.isin(cols)]
        print(temp_Lasso)
        temp_Lasso_df = temp_Lasso[
            ["1- R^2", "1- Adjusted R^2", "MAE", "MSE", "RMSE", "Percent_Error", "sigmoid % Error",
             "1- Accuracy"]].copy()
        temp_Lasso['alpha'] = temp_Lasso['alpha'].astype(str)
        tempX_Lasso = temp_Lasso['alpha']
        plt.plot(tempX_Lasso, temp_Lasso_df)
        plt.xlabel('Linear Regression Lasso',fontsize=12)
        #plt.ylabel('Performance Metrics')
        #plt.title('Performance Evaluation')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        #plt.legend(temp_Lasso_df.columns)
        #plt.show()

        print('plotting positive lasso')
        plt.subplot(223)
        type = ['Linear Regressor Positive Lasso']
        temp_Positive_Lasso = temp[temp.Method.isin(type)]

        cols = [0,0.5,0.75,1.0,5,10,100,500,1000,10000]
        temp_Positive_Lasso = temp_Positive_Lasso[temp_Positive_Lasso.alpha.isin(cols)]
        print(temp_Positive_Lasso)
        temp_Positive_Lasso_df = temp_Positive_Lasso[
            ["1- R^2", "1- Adjusted R^2", "MAE", "MSE", "RMSE", "Percent_Error", "sigmoid % Error",
             "1- Accuracy"]].copy()
        temp_Positive_Lasso['alpha'] = temp_Positive_Lasso['alpha'].astype(str)
        tempX_Positive_Lasso = temp_Positive_Lasso['alpha']
        plt.plot(tempX_Positive_Lasso, temp_Positive_Lasso_df)
        plt.xlabel('Linear Regression Positive Lasso',fontsize=12)
        plt.ylabel('Performance Metrics',fontsize=12)
        #plt.title('Performance Evaluation')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        #plt.legend(temp_Positive_Lasso_df.columns)
        #plt.show()

        plt.subplot(224)
        print('plotting Linear Regressor Elastic Net')
        type = ['Linear Regressor Elastic Net']
        temp_Elastic_Net = temp[temp.Method.isin(type)]
        print(temp_Elastic_Net)
        temp_Elastic_Net_df = temp_Elastic_Net[
            ["1- R^2", "1- Adjusted R^2", "MAE", "MSE", "RMSE", "Percent_Error", "sigmoid % Error",
             "1- Accuracy"]].copy()
        temp_Elastic_Net['alpha'] = temp_Elastic_Net['alpha'].astype(str)
        tempX_Elastic_Net= temp_Elastic_Net['alpha']
        plt.plot(tempX_Elastic_Net, temp_Elastic_Net_df)
        plt.xlabel('Linear Regressor Elastic Net',fontsize=12)
        #plt.ylabel('Performance Metrics')
        #plt.title('Performance Evaluation')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.legend(temp_Elastic_Net_df.columns)
        plt.show()





    def findingloops(self, X, Y):
        print("inside finding loops")
        rows = X.shape[0]
        cols = X.shape[1]
        listcolumns = X.columns
        print("columns are ")
        print(listcolumns)
        cols1=[]
        for i in range(0, cols):
            cols1.append(i)

        self.callingMLModels(X, Y, listcolumns[cols1])
        print("Finishing Loop")
        self.create_ML_Error_csv()

    def callingMLModels(self, X, Y, colslist):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
        list_of_columns = X.columns

        list_ofalpha=[-10,0,0.5,0.75,1.0,5,10,100,500,1000,10000]


        self.LinearRegression1(X_train, X_test, Y_train, Y_test, list_of_columns, colslist)

        for i in list_ofalpha:
            print('alpha is ',i)
            self.LinearRegression_Ridge(X_train, X_test, Y_train, Y_test, list_of_columns, colslist,i)
            self.LinearRegression_Lasso(X_train, X_test, Y_train, Y_test, list_of_columns, colslist,i)
            self.LinearRegression_Positive_Lasso(X_train, X_test, Y_train, Y_test, list_of_columns, colslist,i)
            self.LinearRegression_Elastic_Net(X_train, X_test, Y_train, Y_test, list_of_columns, colslist,i)


        # self.XGBoost_Regressor(X_train, X_test, Y_train, Y_test, colslist)
        # self.RandomRegressor(X_train, X_test, Y_train, Y_test, colslist)
        # self.KNN(X_train, X_test, Y_train, Y_test, colslist)
        # self.SVM(X_train, X_test, Y_train, Y_test, colslist)
        # self.MLPRegressor(X_train, X_test, Y_train, Y_test, colslist)
        # self.print_ML_errors()

    def create_ML_Error_csv(self):
        print("copying the dataframe to a new csv file")
        # print(list(self.df_combined_file['ZIP_OR_POSTAL_CODE'].unique()))

        self.df_ML_errors.to_csv(path + "ML Errors Linear with alpha.csv", index=False)

    def LinearRegression_Ridge(self, X_train, X_test, Y_train, Y_test, list_of_columns, colslist,alpha_val):
        print('---------------------------------------------')
        print('LinearRegression Ridge')
        # values converts it into a numpy array
        x_train = X_train[list_of_columns]
        x_test = X_test[list_of_columns]

        y_train = Y_train
        y_test = Y_test

        linear_regressor = Ridge(fit_intercept=True,alpha=alpha_val)  # create object for the class
        linear_regressor.fit(x_train, y_train)  # perform linear regression
        Y_pred = linear_regressor.predict(x_test)  # make predictions
        print('---------------------------------------------')
        print('Coeff :', linear_regressor.coef_)
        print('Intercept', linear_regressor.intercept_)
        print('LScore', linear_regressor.score(x_test, y_test))

        # Commented to reject Train data evaluation
        '''

        print('Evaluation of Train Data')
        print('---------------------------------------------')
        y_pred_train = linear_regressor.predict(x_train)
        self.FindErrors(x_train, y_train, y_pred_train, 'Linear Regressor Train')
        '''

        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = linear_regressor.predict(x_test)
        # Model Evaluation
        self.FindErrors(x_test, y_test, y_test_pred, 'Linear Regressor Ridge', colslist,alpha_val)

    def LinearRegression_Elastic_Net(self, X_train, X_test, Y_train, Y_test, list_of_columns, colslist,alpha_val):
        print('---------------------------------------------')
        print('LinearRegression Elastic Net')
        # values converts it into a numpy array
        x_train = X_train[list_of_columns]
        x_test = X_test[list_of_columns]

        y_train = Y_train
        y_test = Y_test

        linear_regressor = ElasticNet(fit_intercept=True,alpha=alpha_val)  # create object for the class
        linear_regressor.fit(x_train, y_train)  # perform linear regression
        Y_pred = linear_regressor.predict(x_test)  # make predictions
        print('---------------------------------------------')
        print('Coeff :', linear_regressor.coef_)
        print('Intercept', linear_regressor.intercept_)
        print('LScore', linear_regressor.score(x_test, y_test))

        # Commented to reject Train data evaluation
        '''

        print('Evaluation of Train Data')
        print('---------------------------------------------')
        y_pred_train = linear_regressor.predict(x_train)
        self.FindErrors(x_train, y_train, y_pred_train, 'Linear Regressor Train')
        '''

        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = linear_regressor.predict(x_test)
        # Model Evaluation
        self.FindErrors(x_test, y_test, y_test_pred, 'Linear Regressor Elastic Net', colslist,alpha_val)

    def LinearRegression_Positive_Lasso(self, X_train, X_test, Y_train, Y_test, list_of_columns, colslist,alpha_val):
        print('---------------------------------------------')
        print('LinearRegression Positive Lasso')
        # values converts it into a numpy array
        x_train = X_train[list_of_columns]
        x_test = X_test[list_of_columns]

        y_train = Y_train
        y_test = Y_test

        linear_regressor = Lasso(fit_intercept=True,positive=False,alpha=alpha_val)  # create object for the class
        linear_regressor.fit(x_train, y_train)  # perform linear regression
        Y_pred = linear_regressor.predict(x_test)  # make predictions
        print('---------------------------------------------')
        print('Coeff :', linear_regressor.coef_)
        print('Intercept', linear_regressor.intercept_)
        print('LScore', linear_regressor.score(x_test, y_test))

        # Commented to reject Train data evaluation
        '''

        print('Evaluation of Train Data')
        print('---------------------------------------------')
        y_pred_train = linear_regressor.predict(x_train)
        self.FindErrors(x_train, y_train, y_pred_train, 'Linear Regressor Train')
        '''

        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = linear_regressor.predict(x_test)
        # Model Evaluation
        self.FindErrors(x_test, y_test, y_test_pred, 'Linear Regressor Positive Lasso', colslist,alpha_val)


def main():
    print("inside Main")
    obj = MLModels()
    obj.set_dir(path)

    obj.load_combined_data("AdditionalDataAndHouseData.csv")

    # Finding Standardisation
    obj.standardise_data()

    # Finding correlation
    # print(obj.correlation_plot_combined_file())

    X, Y = obj.removePrice()
    start_time = time.time()

    obj.findingloops(X, Y)
    print("--- %s seconds for running all loops ---" % (time.time() - start_time))

    obj.plot_ML_errors()


if __name__ == '__main__':
    main()
