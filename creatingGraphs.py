import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import matplotlib.ticker as ticker
import math

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
    params = {'legend.fontsize': 10,
              'legend.handlelength': 1}
    plt.rcParams.update(params)

    def __init__(self):
        self.df_add_house_data_file = {}
        # self.df_ML_errors={}
        self.df_ML_errors = pd.DataFrame(columns=["Method", "R^2", "Adjusted R^2", "MAE", "MSE", "RMSE","HuberLoss",
             "logcosh", "Percent_Error"         ])

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

        #plotting QQ plot
        # data_points = self.df_add_house_data_file[['PRICE']]
        #
        # sm.qqplot(data_points, line='45')
        # py.show()

    def creategraphs(self):
        # Histogram of Beds and Baths and 1p PRICE
        self.df_add_house_data_file['PRICE'].plot.hist()
        plt.title('Histogram of Price')
        plt.xlabel('Price of House (Value in Million Dollars)')
        plt.ylabel('Number of houses')
        #plt.xlim([0, 3*1000000])

        plt.show()

        # self.df_add_house_data_file['BATHS'].plot.hist()
        # plt.title('Histogram of Baths')
        # plt.xlabel('Number of Baths in a house')
        # plt.ylabel('Number of Houses')
        # plt.show()
        #plt.boxplot(self.df_add_house_data_file['BATHS'])
        ax=sns.boxplot(x=self.df_add_house_data_file['BATHS'], y=self.df_add_house_data_file['PRICE'], data=pd.melt(self.df_add_house_data_file))
        #plt.xlim(0,20)
        #plt.xticks(fontsize=10)
        # for ind, label in enumerate(ax.get_xticklabels()):
        #     if ind % 2 == 0:  # every 10th label is kept
        #         label.set_visible(True)
        #     else:
        #         label.set_visible(False)
        # for label in ax.get_xticklabels():
        #     print(np.int(float(label.get_text())) % 1 != 0)
        #     print(np.int(float(label.get_text()))%1!=0)
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.title('Distribution of house prices based on number of baths')
        plt.xlabel('Number of Baths')
        plt.ylabel('Price of house (Value in Million Dollars)')
        plt.xticks(fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.show()
        #
        # self.df_add_house_data_file['BEDS'].plot.hist()
        # plt.title('Histogram of Beds')
        # plt.xlabel('Number of Baths in a house')
        # plt.ylabel('Number of Houses')
        # plt.show()

        ax = sns.boxplot(x=self.df_add_house_data_file['BEDS'], y=self.df_add_house_data_file['PRICE'],    data=pd.melt(self.df_add_house_data_file))
        plt.title('Distribution of house prices based on number of beds')
        plt.xlabel('Number of Beds')
        plt.ylabel('Price of house (Value in Million Dollars)')
        #plt.xticks(fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.show()



        # ax = sns.boxplot(x=self.df_add_house_data_file['BEDS'], y=self.df_add_house_data_file['PRICE'])
        # plt.title('Distribution of house prices based on number of Beds')
        # plt.legend(labels=self.df_add_house_data_file.BEDS.unique())
        #plt.show()

        print('Default')
        corr = self.df_add_house_data_file.corr()
        corr = corr.round(2)
        print(corr)
        print(type(corr))
        '''

        print('Pearson')
        corr = self.df_add_house_data_file.corr(method='pearson')
        corr = corr.round(2)
        print(corr)
        print(type(corr))
        

        print('Kendall')
        corr = self.df_add_house_data_file.corr(method='kendall')
        corr = corr.round(2)
        print(corr)
        print(type(corr))

        print('spearman')
        corr = self.df_add_house_data_file.corr(method='spearman')
        corr = corr.round(2)
        print(corr)
        print(type(corr))
        '''

        # temp=self.df_add_house_data_file
        #
        # print(temp.query('BEDS==22'))

        print('Printing correlation ')
        self.correlation_plot_combined_file()

        print('Graph between Price and CIty ')
        # ax = sns.boxplot(x=self.df_add_house_data_file['CITY'], y=self.df_add_house_data_file['PRICE'],
        #                  data=pd.melt(self.df_add_house_data_file))
        # plt.title('Distribution of house prices based on City')
        # plt.xlabel('Name of City')
        # plt.ylabel('Price of house (Value in Million Dollars)')
        # # plt.xticks(fontsize=12)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # plt.show()

        print('Graph of histogram of log of prices')
        self.df_add_house_data_file['PRICE'] = np.log(self.df_add_house_data_file['PRICE'])
        self.df_add_house_data_file['PRICE'].plot.hist()
        plt.title('Histogram of Price')
        plt.xlabel('Price of House (Value in Million Dollars)')
        plt.ylabel('Number of houses')

        plt.show()
        print(5**2)


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
                print(cols1," and",cols2)
                print(columns[cols1 + cols2])
                print(type(columns[cols1 + cols2]))

            cols1=[]
            cols2=[]

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


if __name__ == '__main__':
    main()