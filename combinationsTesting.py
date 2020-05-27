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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.naive_bayes import GaussianNB

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/Additional Data/'


class combinationTesting:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 2000)
    params = {'legend.fontsize': 10,
              'legend.handlelength': 1}
    plt.rcParams.update(params)

    def __init__(self):
        self.df_add_house_data_file = {}
        # self.df_ML_errors={}
        self.df_ML_errors = pd.DataFrame(columns=["Method", "R^2", "Adjusted R^2", "MAE", "MSE", "RMSE"])

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

        #self.df_add_house_data_file =self.df_add_house_data_file.sample(5)

        print(self.df_add_house_data_file.head())
        print(self.df_add_house_data_file.shape)
        print(self.df_add_house_data_file.columns)

    def LinearRegression1(self, X_train, X_test, Y_train, Y_test, list_of_columns):
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

        print('Evaluation of Train Data')
        print('---------------------------------------------')
        y_pred_train = linear_regressor.predict(x_train)
        self.FindErrors(x_train, y_train, y_pred_train, 'Linear Regressor Train')

        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = linear_regressor.predict(x_test)
        # Model Evaluation
        self.FindErrors(x_test, y_test, y_test_pred, 'Linear Regressor Test')
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

        # print("preprocessing.scale")
        standardized_X = preprocessing.scale(numericvals)
        # print(standardized_X)

        print("preprocessing.scaler")
        print(numericvals)
        scaler = preprocessing.StandardScaler()
        scaler.fit(numericvals)
        scaled = scaler.transform(numericvals)
        print(scaled)

        self.Mean=numericvals['PRICE'].mean()
        print("mean is ",self.Mean)
        self.std =np.std(numericvals['PRICE'])
        print("STD is ", self.std)

        # print(standardized_X)
        # print(standardized_X.dtype)
        print(numericvals.columns.values)

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

    def findpercentCount(self, true_value, pred, percent):
        print("Finding percent count")

        #
        print(true_value.shape)
        print(true_value)
        print(pred)
        print(type(true_value))
        print(type(pred))

        true_value=true_value.to_numpy()
        print(type(true_value))
        print(pred.shape[0])
        print('converting to inverse standardisation')
        count=0
        for row in range(pred.shape[0]):
            #print(row)
            y_truevalue=(true_value[row]*self.std )+self.Mean
            y_predvalue = (pred[row] * self.std) + self.Mean
            #print(y_truevalue," ",y_predvalue,"  from ",true_value[row],"  ",pred[row])
            percentvalue=(y_truevalue*percent)/100

            diff=abs(y_truevalue-y_predvalue)
            if(diff>percentvalue):
                count +=1
            #print(diff)
        return count


    def FindErrors(self, x_value, y_value, y_value_pred, method):
        acc_linreg = metrics.r2_score(y_value, y_value_pred)
        adjusted_r2 = 1 - (1 - metrics.r2_score(y_value, y_value_pred)) * (len(y_value) - 1) / (
                    len(y_value) - x_value.shape[1] - 1)
        mae = metrics.mean_absolute_error(y_value, y_value_pred)
        mse = metrics.mean_squared_error(y_value, y_value_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_value, y_value_pred))
        accuracy=round((acc_linreg * 100.0),2)

        delta=1.5

        #huberLoss=self.huber(y_value,y_value_pred,delta)

        #logcosh=self.logcosh(y_value,y_value_pred)

        percent=25

        count_of_error_prediction=self.findpercentCount(y_value,y_value_pred,percent)

        percent_error=count_of_error_prediction/y_value_pred.shape[0]

        print('more than',percent,' percent error:', percent_error ,y_value_pred.shape[0])


        # f1score=metrics.f1_score(y_value,y_value_pred)
        print('R^2:', acc_linreg)
        print('Adjusted R^2:', adjusted_r2)
        print('MAE:', mae)
        print('MSE:', mse)
        print('RMSE:', rmse)
        #print('huberLoss:  ' ,huberLoss)
        #print('logcosh: ',logcosh)
        print('Accuracy:  %.2f%%' % accuracy)

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
        #plt.show()
        return corr

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


def main():
    print("inside Main")
    obj = combinationTesting()
    obj.set_dir(path)

    obj.load_combined_data("AdditionalDataAndHouseData.csv")
    obj.standardise_data()

    X, Y = obj.removePrice()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

    list_of_columns = X.columns

    obj.LinearRegression1(X_train, X_test, Y_train, Y_test, list_of_columns)
if __name__ == '__main__':
    main()
