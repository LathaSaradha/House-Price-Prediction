print("Hello World")

import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
#from matplotlib import pyplot as plt

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/'


class firstTest:

    def load_Data(filename):
        desired_width = 320

        pd.set_option('display.width', desired_width)

        pd.set_option('display.max_columns', 250)
        print(path + filename)
        print("reading file")
        data = pd.read_csv(path + filename)
        # print(data)
        print(data.head())
        print("Dimensions are")
        print(data.shape)

        # Dropping the unwanted columns
        to_drop = ['SALE TYPE',
                   'SOLD DATE', 'LOT SIZE','$/SQUARE FEET',
                   'DAYS ON MARKET','NEXT OPEN HOUSE END TIME','STATUS','j',
'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)','SOURCE','FAVORITE',
                                                                                             'MLS#','INTERESTED','LATITUDE','LONGITUDE',
                   'HOA/MONTH']

        data.drop(columns=to_drop, inplace=True)
        print("reading file 2")
        print(data.head())
        print('--------------------------------------------------------------------')
        print('Dropping empty content rows')
        # data.dropna(axis="ZIP OR POSTAL CODE")

        # data=data.dropna(axis=0, how='any', thresh=1, subset=None, inplace=False)
        data = data.dropna()

        print("Dimensions after dropping are")
        print(data.shape)
        print(data.head())

        # finding the numeric columns

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        numericvals = data.select_dtypes(include=numerics)
        print('--------------------------------------------------------------------')
        print('Printing numeric vals')
        print(numericvals)



        numericvals.drop(columns='PRICE', inplace=True)
        numericvals.drop(columns='ZIP OR POSTAL CODE', inplace=True)
        print(numericvals)
        #numericvals.drop(columns='ZIP OR POSTAL CODE', inplace=True)
        # Normalising the data
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
        print("Standardisation")
        standardized_X = preprocessing.scale(numericvals)
       # print(standardized_X)
        standardized_dataset = pd.DataFrame(
            { 'BEDS': standardized_X[:, 0], 'BATHS': standardized_X[:, 1],
             'SQUARE FEET': standardized_X[:, 2],'Year Built' : standardized_X[:, 3]  })
        #print(standardized_dataset)

        print('--------------------------------------------------------------------')
        print('Correlation Matrix of Standardized Dataset')

        corr__std_matrix = standardized_dataset.corr()

        print(corr__std_matrix)
        #corr__std_matrix.drop(columns='BATHS', inplace=True)
        corr__std_matrix = corr__std_matrix.dropna()

       # print('after dropping baths')

        print(corr__std_matrix)

        eig_vals_std, eig_vecs_std = np.linalg.eig(corr__std_matrix)

        print("Eigen values of standardized")
        print(eig_vals_std)


        print ("Median of eigen values")
        median=np.median(eig_vals_std)
        print(median)

        '''new_array_col_eigen=np.concatenate((standardized_dataset.columns.values,eig_vals_std),axis=0)
        print(new_array_col_eigen) '''


        new_array=np.vstack([standardized_dataset.columns.values,eig_vals_std])
        print(new_array)

        print('--------------------------------------------------------------------')

        print("Features with eigen values >= median")

        for i in range(0,len(new_array[0])):
            if(new_array[1][i] >=median):
                print(new_array[0][i])


        print('--------------------------------------------------------------------')
        print("PCA matrix with components 2")

        X_std = StandardScaler().fit_transform(numericvals)
        sklearn_pca = sklearnPCA(n_components=2)
        Y_sklearn = sklearn_pca.fit_transform(X_std)

        print("Printing year")

        print(type(numericvals))

        print(numericvals['YEAR BUILT'])

        #print(Y_sklearn)

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



#        print("Eigen vectors of standardized")
#       print(eig_vecs_std)


def main():
    print("inside Main")
    firstTest.load_Data('0 to 175K.csv')
    print("End of Main")


if __name__ == '__main__':
    main()





