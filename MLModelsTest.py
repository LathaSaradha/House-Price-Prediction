import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
sns.set(font_scale=0.5)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/Additional Data/'

class MLModels:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 2000)


    # def __init__(self):
    #     df_crime_file = {}
    #     df_combined_file = {}
    #     df_school_rating={}
    #     df_school_location={}
    #     df_combined_school_data={}
    #     df_complaints_data={}
    #     df_population={}

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

        print(self.df_add_house_data_file.head())
        print(self.df_add_house_data_file.shape)
        print(self.df_add_house_data_file.columns)

    def LinearRegression1(self,  X_train, X_test, Y_train, Y_test,list_of_columns):

        # values converts it into a numpy array
        x_train = X_train[list_of_columns]
        x_test=X_test[list_of_columns]

        y_train=Y_train
        y_test=Y_test


        linear_regressor = LinearRegression(fit_intercept=True)  # create object for the class
        linear_regressor.fit(x_train, y_train)  # perform linear regression
        Y_pred = linear_regressor.predict(x_test)  # make predictions
        print('---------------------------------------------')
        print('Coeff :', linear_regressor.coef_)
        print('Intercept', linear_regressor.intercept_)
        print('LScore', linear_regressor.score(x_test, y_test))

        print('Evaluation of Train Data')
        print('---------------------------------------------')
        y_pred_train = linear_regressor.predict(x_train)
        self.FindErrors(x_train,y_train,y_pred_train)


        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = linear_regressor.predict(x_test)
        # Model Evaluation
        self.FindErrors(x_test, y_test, y_test_pred)


        plt.figure(figsize=(20, 20))
        plt.plot([1, 2, 3])
        plt.subplot(221)
        plt.scatter(y_test, Y_pred)
        plt.xlabel("Prices")
        plt.ylabel("Predicted prices")
        plt.title("Prices vs Predicted prices")


        plt.subplot(222)
        plt.scatter(Y_pred, y_test - Y_pred)
        plt.title("Predicted vs residuals")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")


        # Checking Normality of errors
        plt.subplot(223)
        self.histogram_Residuals(y_test - Y_pred)
        plt.show()

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

        return numericvals,Y

    def FindErrors(self, x_value, y_value, y_value_pred):
        acc_linreg = metrics.r2_score(y_value, y_value_pred)
        print('R^2:', acc_linreg)
        print('Adjusted R^2:',
              1 - (1 - metrics.r2_score(y_value, y_value_pred)) * (len(y_value) - 1) / (len(y_value) - x_value.shape[1] - 1))
        print('MAE:', metrics.mean_absolute_error(y_value, y_value_pred))
        print('MSE:', metrics.mean_squared_error(y_value, y_value_pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_value, y_value_pred)))

    def histogram_Residuals(self,axis):
        plt.hist(axis)
        plt.title("Histogram of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")

    def RandomRegressor(self, X_train, X_test, Y_train, Y_test):
        print('---------------------------------------------')
        print('RandomRegressor')
        reg = RandomForestRegressor()

        # Train the model using the training sets
        reg.fit(X_train, Y_train)
        print('Evaluation of Train Data')
        y_train_pred = reg.predict(X_train)

        self.FindErrors(X_train, Y_train, y_train_pred)
        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = reg.predict(X_test)
        print(Y_test)
        print(y_test_pred)

        self.FindErrors(X_test, Y_test, y_test_pred)

    def XGBoost_Regressor(self, X_train, X_test, Y_train, Y_test):
        print('---------------------------------------------')
        print('XGBoost_Regressor')
        reg = XGBRegressor()

        # Train the model using the training sets
        reg.fit(X_train, Y_train)
        print('Evaluation of Train Data')
        y_train_pred = reg.predict(X_train)
        self.FindErrors(X_train, Y_train, y_train_pred)

        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = reg.predict(X_test)
        self.FindErrors(X_test, Y_test, y_test_pred)

    def standardise_data(self):
        print("Standardisation Step 1")

        numeric = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        numerical = self.df_add_house_data_file.select_dtypes(include=numeric)

        print("Numeric columns are.....")

        numerical = numerical.reset_index(drop=True)
        print(numerical)

        # Standardising the data
        print('--------------------------------------------------------------------')
        print("Standardisation Step 2")

        self.df_add_house_data_file = self.FindStandardizedDataset(numerical)
        print('--------------------------------------------------------------------')
        print('Correlation Matrix of Standardized Dataset')

        Y =  self.df_add_house_data_file['PRICE']

        to_drop2 = ['PRICE']

        print(self.df_add_house_data_file.columns)
        # the place where warning comes
        self.df_add_house_data_file=self.df_add_house_data_file.drop(['PRICE'], axis=1)

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

        print(standardized_X)

        std= pd.DataFrame(data=standardized_X,columns=numericvals.columns.values)
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
        corr=corr.round(2)
        #print(corr)
        #print(type(corr))
        # commented to reduce prints
        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        plt.show()
        return corr


def main():
    print("inside Main")
    obj = MLModels()
    obj.set_dir(path)

    obj.load_combined_data("AdditionalDataAndHouseData.csv")

    # Finding Standardisation
    obj.standardise_data()

    # Finding correlation
    print(obj.correlation_plot_combined_file())

    X, Y = obj.removePrice()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

    list_of_columns=X.columns

    obj.LinearRegression1(X_train, X_test, Y_train, Y_test, list_of_columns)

    obj.XGBoost_Regressor(X_train, X_test, Y_train, Y_test)

    obj.RandomRegressor(X_train, X_test, Y_train, Y_test,)



if __name__ == '__main__':
    main()