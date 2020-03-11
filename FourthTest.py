print("Hello World")

import pandas as pd
from sklearn import preprocessing
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
#from matplotlib import pyplot as plt

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/'


class FourthTest:

    def load_Data(self,filename):

        os.chdir(path)
        desired_width = 320

        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
        # export to csv
        combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')

        pd.set_option('display.width', desired_width)

        pd.set_option('display.max_columns', 250)
        print(path + filename)
        print("reading file")
        data = pd.read_csv(path + filename,low_memory=False)
        # print(data)
        print(data.head())
        print("Dimensions are")
        print(data.shape)

        # Dropping the unwanted columns
        to_drop = ['SALE TYPE',
                   'SOLD DATE', 'LOT SIZE','$/SQUARE FEET',
                   'DAYS ON MARKET','NEXT OPEN HOUSE END TIME','STATUS','j',
'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)','SOURCE','FAVORITE',
                                                                                             'MLS#','INTERESTED',
                   'HOA/MONTH','NEXT OPEN HOUSE START TIME',' ']

        data.drop(columns=to_drop, inplace=True,errors='ignore')

        print("reading file 2")
        print(data.shape)
        print(data.head())
        print(data.columns.values)
        print('--------------------------------------------------------------------')
        print('Dropping empty content rows')

        dataDropped = data.dropna()

        print("Dimensions after dropping are")
        print(dataDropped.shape)
        print(dataDropped.head())

        # finding the numeric columns

        numeric = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        dataDropped[['ZIP OR POSTAL CODE']]=dataDropped[['ZIP OR POSTAL CODE']].astype(float)
        numerical = dataDropped.select_dtypes(include=numeric)
        print('--------------------------------------------------------------------')
        print('Printing numeric vals')
        print(numerical)


        # price=numericvals['PRICE']

        numerical.drop(columns='PRICE', inplace=True, errors='ignore')


        print(numerical)

        # Standardising the data
        print('--------------------------------------------------------------------')
        print("Standardisation")
        standardized_dataset=self.FindStandardizedDataset(numerical)
        '''
        normalized_X = preprocessing.normalize(numericvals ,norm='l1')
       # normalized_X1 = preprocessing.StandardScaler().fit(numericvals)

        pd.set_option('display.width', desired_width)
        print(normalized_X.shape)
        print(normalized_X)



        print('--------------------------------------------------------------------')
        print('Normalaised matrix is')
        normalized_dataset=pd.DataFrame({'ZIP Code': normalized_X[:, 0], 'PRICE': normalized_X[:, 1],'BEDS': normalized_X[:, 2],
                                         'BATHS': normalized_X[:, 3],'Year Built': normalized_X[:, 4] ,'$/sqft': normalized_X[:, 5]})
        print(normalized_dataset)

        print('--------------------------------------------------------------------')
        print('Normalaised matrix is with std scaler')

        normalized_dataset1= pd.DataFrame(
            {'ZIP Code': normalized_X1[:, 0], 'PRICE': normalized_X1[:, 1], 'BEDS': normalized_X1[:, 2],
             'BATHS': normalized_X1[:, 3], 'Year Built': normalized_X1[:, 4], '$/sqft': normalized_X1[:, 5]})
        print(normalized_dataset1)

        print('--------------------------------------------------------------------')
        print('Correlation Matrix of Normalized Dataset')
        corr_matrix=normalized_dataset.corr()
        print(corr_matrix)

        eig_vals, eig_vecs = np.linalg.eig(corr_matrix)

        print("Eigen values of normalized")
        print(eig_vals)
        print("Eigen vectors of normalized")
        print(eig_vecs)
        # standaridizing data

        '''

        print('--------------------------------------------------------------------')
        print('Correlation Matrix of Standardized Dataset')

        corr__std_matrix = self.findCorrStdMatrix(standardized_dataset)

        # print('after dropping baths')

        print(corr__std_matrix)

        eig_vals_std, eig_vecs_std = np.linalg.eig(corr__std_matrix)

        print("Eigen values of standardized")
        print(eig_vals_std)

        median = self.MedianOfEigenValues(eig_vals_std)

        '''new_array_col_eigen=np.concatenate((standardized_dataset.columns.values,eig_vals_std),axis=0)
        print(new_array_col_eigen) '''


        new_array=np.vstack([standardized_dataset.columns.values,eig_vals_std])
        print(new_array)

        self.findValuesgreaterThanMedian(median, new_array)



        # print('--------------------------------------------------------------------')
        # print("PCA matrix with components 2")
        #
        # X_std = StandardScaler().fit_transform(newNumerical)
        # sklearn_pca = sklearnPCA(n_components=2)
        # Y_sklearn = sklearn_pca.fit_transform(X_std)
        #
        # #print(Y_sklearn)

        '''
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(6, 4))

            plt.imshow(Y_sklearn)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            # plt.zlabel('Price')
            plt.legend(loc='lower center')
            plt.tight_layout()
            plt.show()
        '''

    def findValuesgreaterThanMedian(self, median, new_array):
        print('--------------------------------------------------------------------')
        print("Features with eigen values > median")
        for i in range(0, len(new_array[0])):
            if (new_array[1][i] >= median):
                print(new_array[0][i])

    def FindStandardizedDataset(self, numericvals):

        standardized_X = preprocessing.scale(numericvals)
        print(standardized_X)
        print(standardized_X.dtype)
        standardized_dataset = pd.DataFrame(
            {'ZIPCODE': standardized_X[:, 0], 'BEDS': standardized_X[:, 1], 'BATHS': standardized_X[:, 2],
             'SQUARE FEET': standardized_X[:, 3], 'Year Built': standardized_X[:, 4] ,'LATITUDE':standardized_X[:,5], 'LONGITUDE':standardized_X[:,6]})
        #print(standardized_dataset)
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


#        print("Eigen vectors of standardized")
#       print(eig_vecs_std)


def main():
    print("inside Main")
    obj=FourthTest()

    obj.load_Data('combined_csv.csv')
    print("End of Main")
    #os.remove(path + 'combined_csv.csv')


if __name__ == '__main__':
    main()





