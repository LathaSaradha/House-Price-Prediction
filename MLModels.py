#from Project import ManipulatingCombinedCSVFile as mp
from sklearn import linear_model,datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# Load libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/'
class MLModels:
    filename=''




    def load_Data(self,filename):
        desired_width = 400

        pd.set_option('display.width', desired_width)

        pd.set_option('display.max_columns', 300)
        print(path + filename)
        print("reading file")
        data = pd.read_csv(path + filename, low_memory=False, dtype={'CITY': str})
        print(data.head())
        print("Dimensions are")
        print(data.shape)
        print(data.columns.values)
        col_to_drop={'Unnamed: 0'}
        data=data.drop(columns=col_to_drop)
        print("Dimensions after dropping are")
        print(data.shape)
        return data

    def linearRegression(self,data,xvariable):
        df = data.loc[data['SQUARE FEET'] < 10000]
        # values converts it into a numpy array

        X = df[xvariable].values.reshape(-1, 1)

        Y = df['PRICE'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

        linear_regressor = LinearRegression(fit_intercept=True)  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.xlabel(xvariable)
        # frequency label
        plt.ylabel('PRICE')

        print('Coeff :',linear_regressor.coef_)
        print('Intercept' ,linear_regressor.intercept_)
        print('LScore',linear_regressor.score(X,Y))
        print()
        plt.title('Scatter plot with price and ' + xvariable)
        #plt.legend()
        plt.show()


    '''
    def randomForest(self,train):
        models = []
        results=[]
        names=[]
        models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(gamma='auto')))

        #model = RandomForestClassifier(n_estimators=100,  bootstrap=True, max_features='sqrt')
        train_labels=train.columns.values
        print(train_labels)


        #model.fit(train, train_labels)

        X = train[:, 0:4]
        y = train[:, 4]
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

        for name, model in models:
            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results)
            names.append(name)
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    '''

    def plot(self,data,x,y):
        df = data.loc[data['SQUARE FEET']<10000]
        x_axis= df[x]
        y_axis= df[y]
        plt.scatter(x_axis, y_axis, label="stars", color="green",marker="*", s=30)
        plt.xlabel(x)
        # frequency label
        plt.ylabel(y)
        min =0
        max=3000
        step=100
        #plt.xticks(np.arange(min, max, step))
        plt.xlim()
        plt.grid(True)

        # plot title
        plt.title('Scatter plot with price and '+y)
        plt.legend()

        # function to show the plot
        #plt.show()


    def query(self,data):
        data['SQUARE FEET'] = (data['SQUARE FEET']).astype(float, copy=True)
        print(data.dtypes)
        df = data.loc[data['SQUARE FEET']>7000]
        #print(df)

    def LinearRegression1(self,  X_train, X_test, Y_train, Y_test,list_of_columns):

        # values converts it into a numpy array


        x_train = X_train[list_of_columns]
        x_test=X_test[list_of_columns]

        y_train=Y_train
        y_test=Y_test


        linear_regressor = LinearRegression(fit_intercept=True)  # create object for the class
        linear_regressor.fit(x_train, y_train)  # perform linear regression
        Y_pred = linear_regressor.predict(x_test)  # make predictions




        print('Coeff :', linear_regressor.coef_)
        print('Intercept', linear_regressor.intercept_)
        print('LScore', linear_regressor.score(x_test, y_test))

        print('Evaluation of Train Data')
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


    def histogram_Residuals(self,axis):
        plt.hist(axis)
        plt.title("Histogram of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")

    def scatter_plot(self,Xaxis,Yaxis,title,xlabel,ylabel):
        plt.scatter(Xaxis, Yaxis)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def FindErrors(self, x_value, y_value, y_value_pred):
        acc_linreg = metrics.r2_score(y_value, y_value_pred)
        print('R^2:', acc_linreg)
        print('Adjusted R^2:',
              1 - (1 - metrics.r2_score(y_value, y_value_pred)) * (len(y_value) - 1) / (len(y_value) - x_value.shape[1] - 1))
        print('MAE:', metrics.mean_absolute_error(y_value, y_value_pred))
        print('MSE:', metrics.mean_squared_error(y_value, y_value_pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_value, y_value_pred)))

    def RandomRegressor(self,X_train, X_test, Y_train, Y_test):
        reg = RandomForestRegressor()

        # Train the model using the training sets
        reg.fit(X_train, Y_train)
        print('Evaluation of Train Data')
        y_train_pred = reg.predict(X_train)

        self.FindErrors(X_train,Y_train,y_train_pred)
        print('---------------------------------------------')
        print('Evaluation of Test Data')
        y_test_pred = reg.predict(X_test)

        self.FindErrors(X_test, Y_test, y_test_pred)


    def XGBoost_Regressor(self,X_train, X_test, Y_train, Y_test):
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



def main():
    print("inside Main")
   # obj =mp.Manipulation()
   # obj.load_Data('combined_csv.csv')
    obj1=MLModels()
    print('Loading data')
    data=obj1.load_Data('manipulatedTrain.csv')
    obj1.query(data)

    X = data.drop(['PRICE'], axis=1)
    print(X.head)
    Y = data['PRICE']
    print(Y.head)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)


   # print('Plotting')

    '''
    obj1.plot(train,'SQUARE FEET','PRICE')
    obj1.linearRegression(train,'SQUARE FEET')
    '''



    #obj1.plot(data,'BEDS','PRICE')
    #obj1.linearRegression(data,'BEDS')

    #obj1.linearRegression(data, 'SQUARE FEET')

    #obj1.linearRegression(data, 'BATHS')
   # obj1.linearRegression(data,'ZIP OR POSTAL CODE')
    print(data.head)
    print('Multiple Linear Regression')
    list_of_columns=['SQUARE FEET','ZIPCODE','BEDS','BATHS','Year Built','LATITUDE','LONGITUDE','CITY numeric']
    obj1.LinearRegression1( X_train, X_test, Y_train, Y_test,list_of_columns)
    print('---------------------------------------------')
    print('Random Regression')
    obj1.RandomRegressor(X_train,X_test,Y_train,Y_test)

    print('---------------------------------------------')
    print('XGB Regression')
    obj1.XGBoost_Regressor(X_train, X_test, Y_train, Y_test)



    print("removing BEDS and baths columns")
    new_X = data.drop({'BEDS','BATHS'}, axis=1)
    #print(new_X.head)
    print(new_X.columns.values)
    X_train, X_test, Y_train, Y_test = train_test_split(new_X, Y, test_size=0.2, random_state=4)
    print('---------------------------------------------')
    print('Random Regression')
    obj1.RandomRegressor(X_train,X_test,Y_train,Y_test)

    print('---------------------------------------------')
    print('XGB Regression')
    obj1.XGBoost_Regressor(X_train, X_test, Y_train, Y_test)



if __name__ == '__main__':
    main()