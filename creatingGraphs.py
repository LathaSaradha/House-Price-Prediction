import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import matplotlib.ticker as ticker
import math
import itertools

import statsmodels.api as sm
import pylab as py


sns.set(font_scale=2.0)
from sklearn.linear_model import LinearRegression
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
    params = {'legend.fontsize': 12,
              'legend.handlelength': 1}
    plt.rcParams.update(params)

    def __init__(self):
        self.df_add_house_data_file = {}
        # self.df_ML_errors={}

        self.df_ML_errors = pd.DataFrame(columns=["Method", "R^2", "Adjusted R^2", "MAE", "MSE", "RMSE",
                                            "Percent_Error", "ColsList", "sigmoid % Error"])
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



    def creategraphs(self):
        # Histogram of Beds and Baths and 1p PRICE
        self.df_add_house_data_file['PRICE'].plot.hist()
        plt.title('Histogram of Price')
        plt.xlabel('Price of House (Value in Million Dollars)')
        plt.ylabel('Number of houses')
        #plt.xlim([0, 3*1000000])

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

        # print('Graph of histogram of log of prices')
        # self.df_add_house_data_file['PRICE'] = np.log(self.df_add_house_data_file['PRICE'])
        # self.df_add_house_data_file['PRICE'].plot.hist()
        # plt.title('Histogram of Price')
        # plt.xlabel('Price of House (Value in Million Dollars)')
        # plt.ylabel('Number of houses')
        #
        # plt.show()



    def correlation_plot_combined_file(self):
        print(self.df_add_house_data_file.columns)
        corr = self.df_add_house_data_file.corr()
        corr = corr.round(2)
        print(corr)
        print(type(corr))
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        # commented to reduce prints
        # sns.set(font_scale=0.5)
        # ax = sns.heatmap(
        #     corr,
        #
        #     annot=True,
        #     vmin=-1, vmax=1, center=0,
        #     cmap=sns.diverging_palette(20, 220, n=200),
        #     square=True
        # )
        # plt.show()

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

    def findingloops(self):
        cols1=[1,3,5]
        cols2=[6,7,8]
        columns = self.df_add_house_data_file.columns
        print(self.df_add_house_data_file.shape)
        rows=self.df_add_house_data_file.shape[0]
        cols=self.df_add_house_data_file.shape[1]
        print(rows)
        print(cols)

        temp=self.df_add_house_data_file.iloc[0:rows,cols1+cols2]
        print(temp)
        print(columns[cols1])

        cols1=[]
        cols2=[]
        for i in range(0,cols):
            cols1.append(i)
            for j in range(i+1,cols):
                cols2.append(j)
                # print(cols1," and",cols2)
                # print(columns[cols1 + cols2])
                # print(type(columns[cols1 + cols2]))

            cols1=[]
            cols2=[]

        for i in range(0, cols):
            print('i is ',i)
            print(cols)
            data=itertools.combinations(cols,i)
            subsets = set(data)
            print(subsets)

    def findingLoopsNew(self):
        print("inside finding loops")
        rows = 0
        cols = 26
        listcolumns = self.df_add_house_data_file.columns
        print("columns are ")
        print(listcolumns)
        cols1 = []
        cols2 = []


        for i in range(1,cols):
            print('i is ',i)
            cc=list(itertools.combinations(listcolumns, i))
            print(cc)
            print

        print("Finishing Loop")



    def findSigmoid(self):
        # x = np.array(self.df_add_house_data_file['PRICE'], dtype=np.float64)
        var=-0.5
        x=np.array([0.1,0.5,1.0,1.5,2.0,2.5,3], dtype=np.float64)
        # y = np.exp(var * x) / (1 + np.exp(var * x))

        y = np.array(1 / (1 + np.exp(-x)), dtype=np.float64)
        plt.plot(x, y)
        plt.xlabel('PRICE')
        plt.ylabel('Percent')
        plt.show()

        y1 = np.array(0.3*np.exp(-x), dtype=np.float64)
        plt.plot(x, y1)
        plt.xlabel('PRICE')
        plt.ylabel('Percent')
        plt.show()

        for row in range(x.shape[0]):
            print(x[row])
            y=0.25*math.exp(-x[row])
            print(y)


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

    def create_Linear_graph(self):
        self.standardise_data()
        X, Y = self.removePrice()
        rows = X.shape[0]

        colslist = ['YEAR_BUILT']
        self.call_LinearRegression(Y, colslist, 'set6')

        colslist = ['AGE']
        self.call_LinearRegression(Y, colslist, 'set7')

        colslist = ['Population']
        self.call_LinearRegression(Y, colslist, 'set12')

        colslist = ['ZIP_OR_POSTAL_CODE']
        self.call_LinearRegression(Y, colslist, 'set8')

        colslist = ['Total_Num_ofHospitals']
        self.call_LinearRegression(Y, colslist, 'set13')

        colslist = ['Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount']
        self.call_LinearRegression(Y, colslist, 'set9')



        colslist = ['BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric']
        self.call_LinearRegression(Y, colslist, 'set4')

        colslist = ['Total_Num_of_Subways']
        self.call_LinearRegression(Y, colslist, 'setx')





        colslist = ['Total_Num_of_Subways', 'min_dist_station', 'Num_of_Retail_stores', 'min_dist_retail_store']
        self.call_LinearRegression(Y, colslist, 'set14')




        colslist =            ['Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount', 'Level_C_SchoolCount', 'Level_D_SchoolCount',
             'Level_F_SchoolCount', 'Total_Number_of_Schools', 'Num_Complaints_schools', 'Population', 'People/Sq_Mile',
             'Total_Num_ofHospitals']
        self.call_LinearRegression(Y, colslist, 'set10')


        colslist =            ['Level_F_SchoolCount', 'Total_Number_of_Schools', 'Num_Complaints_schools', 'Population', 'People/Sq_Mile',
             'Total_Num_ofHospitals', 'Total_Num_of_Subways', 'min_dist_station', 'Num_of_Retail_stores',
             'min_dist_retail_store']
        self.call_LinearRegression(Y, colslist, 'set11')

        colslist = ['SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric',
                    'Total_Num_ofComplaints',
                    'Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount', 'Level_C_SchoolCount']
        self.call_LinearRegression(Y, colslist, 'set5')

        colslist =['SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric', 'Total_Num_ofComplaints',
               'Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount', 'Level_C_SchoolCount',
               'Level_D_SchoolCount', 'Level_F_SchoolCount', 'Total_Number_of_Schools', 'Num_Complaints_schools']

        self.call_LinearRegression(Y, colslist, 'seta')

        colslist = ['BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric',
                    'Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount',
                    'Level_C_SchoolCount', 'Level_D_SchoolCount', 'Level_F_SchoolCount', 'Total_Number_of_Schools',
                    'Num_Complaints_schools', 'Population']

        self.call_LinearRegression(Y, colslist, 'sety')

        colslist =['ZIP_OR_POSTAL_CODE', 'BEDS', 'BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE',
               'CITY numeric', 'Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount']

        self.call_LinearRegression(Y, colslist, 'setz')


        colslist = ['ZIP_OR_POSTAL_CODE', 'BEDS', 'BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE',
                    'CITY numeric', 'Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount',
                    'Level_B_SchoolCount',
                    'Level_C_SchoolCount', 'Level_D_SchoolCount', 'Level_F_SchoolCount', 'Total_Number_of_Schools',
                    'Num_Complaints_schools', 'Population', 'People/Sq_Mile', 'Total_Num_ofHospitals',
                    'Total_Num_of_Subways', 'min_dist_station', 'Num_of_Retail_stores', 'min_dist_retail_store']
        self.call_LinearRegression(Y, colslist, 'set1')

        colslist = ['BEDS', 'BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric',
                    'Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount',
                    'Level_C_SchoolCount', 'Level_D_SchoolCount', 'Level_F_SchoolCount', 'Total_Number_of_Schools',
                    'Num_Complaints_schools', 'Population', 'People/Sq_Mile', 'Total_Num_ofHospitals',
                    'Total_Num_of_Subways', 'min_dist_station', 'Num_of_Retail_stores', 'min_dist_retail_store']
        self.call_LinearRegression(Y, colslist, 'set2')

        colslist = ['BEDS', 'BATHS', 'SQUARE_FEET', 'YEAR_BUILT', 'LATITUDE', 'LONGITUDE', 'AGE', 'CITY numeric',
                    'Total_Num_ofComplaints', 'Total_crimes', 'Level_A_SchoolCount', 'Level_B_SchoolCount',
                    'Level_C_SchoolCount', 'Level_D_SchoolCount', 'Level_F_SchoolCount', 'Total_Number_of_Schools',
                    'Num_Complaints_schools', 'Population', 'People/Sq_Mile', 'Total_Num_ofHospitals',
                    'Total_Num_of_Subways', 'min_dist_station', 'Num_of_Retail_stores', 'min_dist_retail_store',
                    'Num_of_Retail_stores_Zipcode']
        self.call_LinearRegression(Y, colslist, 'set3')




        self.plot_ML_errors()
        self.print_ML_errors()

    def call_LinearRegression(self, Y, colslist,set):
        tempX = self.df_add_house_data_file[colslist]
        X_train, X_test, Y_train, Y_test = train_test_split(tempX, Y, test_size=0.2, random_state=4)
        list_of_columns = X_train.columns
        self.LinearRegression1(X_train, X_test, Y_train, Y_test, list_of_columns, colslist, set)

    def plot_ML_errors(self):
        print('Plotting errors')
        temp=self.df_ML_errors[["R^2", "Adjusted R^2", "MAE", "MSE", "RMSE","Percent_Error","sigmoid % Error","Accuracy"]].copy()
        temp['R^2']=1-temp['R^2']
        temp['Adjusted R^2'] = 1 - temp['Adjusted R^2']
        temp['Accuracy'] = 1- (temp['Accuracy']/100)

        temp = temp.rename(columns={"R^2": "1- R^2"})
        temp = temp.rename(columns={"Adjusted R^2": "1- Adjusted R^2"})
        tempX = self.df_ML_errors['Method']
        plt.plot(tempX, temp)
        plt.xlabel('Linear Regression Methods',fontsize=14)
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

    def print_ML_errors(self):
        print(self.df_ML_errors)
        print(self.df_ML_errors.shape)



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

    def findpercentCount(self, true_value, pred, percent):
        print("Finding percent count")

        true_value = true_value.to_numpy()
        print('converting to inverse standardisation')
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



def main():
    print("inside Main")
    obj = MLModels()
    obj.set_dir(path)

    obj.load_combined_data("AdditionalDataAndHouseData.csv")
    print("Calling graphs")
    obj.creategraphs()
    print('Finding Loops')
    #obj.findingloops()

    #print('Finding sigmoid')
    #obj.findSigmoid()
    obj.create_Linear_graph()


if __name__ == '__main__':
    main()