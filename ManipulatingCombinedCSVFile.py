print("Hello World")

import pandas as pd
from sklearn import preprocessing
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
#from matplotlib import pyplot as plt

'''
This program takes the combined_csv.csv file
Then change the column 'CITY' to numeric data and 'PROPERTY TYPE' to numeric data

drops and outlier #Removing an outlier
        index1 = data[(data['ZIP OR POSTAL CODE'] =='07206-1637')].index
        
Finds the standarisation and correlation and then save the excel to 'manipulatedTrain.csv'

'''

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/'

class Manipulation:

    def load_Data(self,filename):
        desired_width = 400

        pd.set_option('display.width', desired_width)

        pd.set_option('display.max_columns', 300)
        print(path + filename)
        print("reading file")
        data = pd.read_csv(path + filename, low_memory=False,dtype={'CITY':str})
        print(data.head())
        print("Dimensions are")
        print(data.shape)
        print("Converting city to string")


        self.changingCityColumn(data)

        self.changePropertyTypeCol(data)
        to_drop = ['SALE TYPE',
                   'SOLD DATE', 'LOT SIZE', '$/SQUARE FEET',
                   'DAYS ON MARKET', 'NEXT OPEN HOUSE END TIME', 'STATUS', 'j',
                   'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)',
                   'SOURCE', 'FAVORITE',
                   'MLS#', 'INTERESTED',
                   'HOA/MONTH', 'NEXT OPEN HOUSE START TIME', ' ']

        data.drop(columns=to_drop, inplace=True, errors='ignore')
        dataDropped = data.dropna()

        print("Dimensions after dropping are")
        print(dataDropped.shape)
        print(dataDropped.head())

        #Removing an outlier
        index1 = data[(data['ZIP OR POSTAL CODE'] =='07206-1637')].index

        data = data.drop(index=index1, inplace=True)


        numeric = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        dataDropped['ZIP OR POSTAL CODE'] = (dataDropped['ZIP OR POSTAL CODE']).astype(float,copy=True)


        numerical = dataDropped.select_dtypes(include=numeric)
        print('--------------------------------------------------------------------')
        print('Printing numeric vals')

        numerical =numerical.reset_index(drop=True)
        print(numerical)


        print(numerical)

        # Standardising the data
        print('--------------------------------------------------------------------')
        print("Standardisation")
        standardized_dataset = self.FindStandardizedDataset(numerical)
        print('--------------------------------------------------------------------')
        print('Correlation Matrix of Standardized Dataset')
        Y = standardized_dataset['PRICE']

        to_drop2 = ['PRICE']

        # the place where warning comes
        numerical.drop(columns=to_drop2, inplace=True, errors='ignore')
        #Calling correlation method
        corr__std_matrix = self.findCorrStdMatrix(standardized_dataset)


        self.calculationofEigenvalues(corr__std_matrix, standardized_dataset)
        print(Y)
        standardized_dataset['PRICE']=Y

        standardized_dataset.to_csv(path+'manipulatedTrain.csv')

    def calculationofEigenvalues(self, corr__std_matrix, standardized_dataset):
        eig_vals_std, eig_vecs_std = np.linalg.eig(corr__std_matrix)

        median = self.MedianOfEigenValues(eig_vals_std)
        new_array = np.vstack([standardized_dataset.columns.values, eig_vals_std])
        print(new_array)
        self.findValuesgreaterThanMedian(median, new_array)

    def changePropertyTypeCol(self, data):
        #print(data['PROPERTY TYPE'].unique())
        clean = {'PROPERTY TYPE': {'Condo/Co-op': 1, 'Single Family Residential': 2, 'Multi-Family (2-4 Unit)': 3,
                                   'Townhouse': 4,
                                   'Multi-Family (5+ Unit)': 5}

                 }
        data.replace(clean, inplace=True)
        #print(data['PROPERTY TYPE'].unique())
        data = data.drop(data[data['PROPERTY TYPE'] == 'Vacant Land'].index)
        data = data.drop(data[data['PROPERTY TYPE'] == 'Unknown'].index)
        # print(data[data['PROPERTY TYPE'] == nan].index)
        data['PROPERTY TYPE'] = data['PROPERTY TYPE'].dropna()
        index1 = data[(data['PROPERTY TYPE'] != 1) & (data['PROPERTY TYPE'] != 2) & (data['PROPERTY TYPE'] != 3) & (
                    data['PROPERTY TYPE'] != 4) & (data['PROPERTY TYPE'] != 5)].index
        #print(index1)
        data = data.drop(index=index1, inplace=True)
        # print(data['PROPERTY TYPE'].unique())

    def changingCityColumn(self, data):
        #print(data.dtypes)
        #print(len(data.CITY.unique()))
        data['CITY'] = data['CITY'].str.upper()
        data['CITY'] = data['CITY'].dropna()
        #print(data.CITY.unique())
        data['CITY numeric'] = data['CITY'].apply(lambda r: self.findCharacterSum(r))
        # print(data['new'])
        #print(len(data['CITY numeric'].unique()))
        #print(len(data.CITY.unique()))

    def findCharacterSum(self,s):
        sum = 0
        if(type(s)!=str):
            return 0
        #print(s)
        #print(len(s))
        for j in range(len(s)):
            sum += ord(s[j])
        #print(sum)
        return sum

    def findValuesgreaterThanMedian(self, median, new_array):
        print('--------------------------------------------------------------------')
        print("Features with eigen values > median")
        for i in range(0, len(new_array[0])):
            if (new_array[1][i] >= median):
                print(new_array[0][i])

    def FindStandardizedDataset(self, numericvals):

        standardized_X = preprocessing.scale(numericvals)
        #print(standardized_X)
        #print(standardized_X.dtype)
        print(numericvals.columns.values)
        #standardized_dataset=pd.DataFrame(standardized_X)
        #print(standardized_dataset)
        #standardized_dataset1= pd.DataFrame(standardized_X.astype("float32").T, columns=numericvals.columns.values).astype({k: v[0] for k, v in standardized_X.dtype.fields.items()})
        #standardized_dataset1=pd.DataFrame(data=standardized_X[1:, 1:],  index = standardized_X[1:, 0], columns = numericvals.columns.values[0, 1:])
        #print(standardized_dataset1)

        standardized_dataset = pd.DataFrame(
            {'ZIPCODE': standardized_X[:, 0],'PRICE':standardized_X[:,1],
             'BEDS': standardized_X[:, 2], 'BATHS': standardized_X[:, 3],
             'SQUARE FEET': standardized_X[:, 4], 'Year Built': standardized_X[:, 5] ,'LATITUDE':standardized_X[:,6], 'LONGITUDE':standardized_X[:,7], 'CITY numeric' :standardized_X[:,8]})
        print(standardized_dataset)
        return standardized_dataset

    def findCorrStdMatrix(self, standardized_dataset):
        corr__std_matrix = standardized_dataset.corr()
        print(corr__std_matrix)

        corr__std_matrix_new = corr__std_matrix.dropna()
        return corr__std_matrix_new

    def MedianOfEigenValues(self, eig_vals_std):
        print("Median of eigen values")
        median = np.median(eig_vals_std)
        print(median)
        return median











def main():
    print("inside Main")
    obj = Manipulation()
    obj.load_Data('combined_csv.csv')


if __name__ == '__main__':
    main()