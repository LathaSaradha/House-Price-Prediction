'''
Author : Latha Saradha
Purpose : This file is used to load the CSV files which contain the data for the house
 and perform data pre-processing for the house features.
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os

path= pathlib.Path().absolute()/"Data"

class cleanData:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    def __init__(self):
        df_combined_file = {}

    # Setting the working directory
    def set_dir(self,path):
        try:
            os.chdir(path)
            print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")

        print(os.getcwd())

    # Laoding the data file for House
    def loadData(self, filename):
        print('Reading', filename)
        print(type(path))
        print(path)
        self.df_combined_file = pd.read_csv(filename,index_col=False)

    def identify_columns_with_missing_values(self):

        # Find the number of rows with empty values in all columns and dropping those
        self.df_combined_file.dropna(axis=0, how='all',inplace=True)
        print('Dimensions after removal are ',self.df_combined_file.shape)

    def uniqueValues(self):
        print ('Unique values')
        print(list(self.df_combined_file['SALE TYPE'].unique()))
        print(list(self.df_combined_file['PROPERTY TYPE'].unique()))
        print(list(self.df_combined_file['STATUS'].unique()))
        print(list(self.df_combined_file['LOCATION'].unique()))

        print((len
               (list(self.df_combined_file['LOCATION'].unique()))
               ))
        print(list(self.df_combined_file['CITY'].unique()))

        print((len
               (list(self.df_combined_file['CITY'].unique()))
               ))


    def corr_plots_price_lat_long(self):
        corr = self.df_combined_file[['PRICE','LATITUDE','LONGITUDE']].corr()
        print(corr)
        # Heat Map distribution for the Latitude and Longitude with Price.
        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        plt.title('Correlation between Price and Lat/Long')
        plt.show()

    # Method to find the Heat Map distribution for the Price and Year Built
    def data_price_yearbuilt(self):
        temp_df=self.df_combined_file[['PRICE','YEAR BUILT']]
        temp_df1=temp_df.dropna()

        corr = temp_df1.corr()
        print(corr)


        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        plt.title('Correlation between Price and Year Built')
        plt.show()

    # Method to convert the year built feature to age for easy calculation.
    def data_price_age(self):
        temp_df = self.df_combined_file[['PRICE', 'YEAR BUILT']]
        temp_df1 = temp_df.dropna()
        minYear=min(temp_df1['YEAR BUILT'])
        maxYear = max(temp_df1['YEAR BUILT'])
        print("min Year is " ,minYear)
        print("max Year is ", maxYear)
        print(list(temp_df1['YEAR BUILT'].unique()))
        temp_df1.columns = [column.replace(" ", "_") for column in temp_df1.columns]
        temp_df1=temp_df1.loc[temp_df1['YEAR_BUILT'] <1800.0 , ['YEAR_BUILT']]

        years=[1060,1055,1500,1025]
        self.df_combined_file.columns = [column.replace(" ", "_") for column in self.df_combined_file.columns]
        print(self.df_combined_file.query('YEAR_BUILT==1060.0'))

        # creating additional column to the dataframe
        self.df_combined_file['AGE']=2021-self.df_combined_file['YEAR_BUILT']

        self.df_combined_file = self.df_combined_file[self.df_combined_file.YEAR_BUILT >= 1800.0]
        #self.df_combined_file =self.df_combined_file.reset_index(drop=True)

        temp_df = self.df_combined_file[['PRICE', 'AGE']]
        temp_df1 = temp_df.dropna()

        corr = temp_df1.corr()
        print(corr)

    # Method to filter the necessary property types
    def clean_PropertyType(self):

        print()
        print(list(self.df_combined_file['PROPERTY_TYPE'].unique()))
        propertyTypes = ['Condo/Co-op', 'Single Family Residential', 'Multi-Family (2-4 Unit)', 'Townhouse','Multi-Family (5+ Unit)']
        self.df_combined_file = self.df_combined_file [self.df_combined_file.PROPERTY_TYPE.isin(propertyTypes)]
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)
        print(list(self.df_combined_file['PROPERTY_TYPE'].unique()))


        propertyTypes_Multi=['Multi-Family (2-4 Unit)', 'Multi-Family (5+ Unit)']
        new_datafram_MultiFamily=self.df_combined_file[self.df_combined_file.PROPERTY_TYPE.isin(propertyTypes_Multi)]

    # Method to keep the necessary columns and delete the unwanted columns
    def keep_necessary_columns(self):
        print(self.df_combined_file.shape)
        print(self.df_combined_file.columns)
        col_to_drop=['SALE_TYPE','SOLD_DATE','ADDRESS','DAYS_ON_MARKET','NEXT_OPEN_HOUSE_START_TIME','INTERESTED','FAVORITE','SOURCE','MLS#','SOURCE','LOT_SIZE','$/SQUARE_FEET','HOA/MONTH', 'STATUS', 'j','NEXT_OPEN_HOUSE_END_TIME','_']
        self.df_combined_file=self.df_combined_file.drop(labels=col_to_drop,axis=1)
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)

    # Method to remove the duplicate records
    def remove_duplicates(self):
        print('Before removing duplicates')
        print(self.df_combined_file.shape)
        self.df_combined_file.drop_duplicates(keep='first',inplace=True,ignore_index=True)
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)
        print('After removing duplicates')
        print(self.df_combined_file.shape)

    # Method to find the BEDS calculation removing rows where there are no value of BEDS
    def beds_Baths(self):

        self.df_combined_file=self.df_combined_file [~self.df_combined_file.BEDS.isna()]
        print(self.df_combined_file)

        # Find calculations for Baths removing rows where there are no value of BATHS
        print('After checking BATHS')
        print(list(self.df_combined_file['BATHS'].unique()))
        self.df_combined_file = self.df_combined_file[~self.df_combined_file.BATHS.isna()]
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)
        print(self.df_combined_file.shape)

        print(list(self.df_combined_file['BEDS'].unique()))

    # Method to impute the value for records with no sqft value and fill with the mean value of beds and baths
    def clean_sqft(self):
        # creating a new dataframe with data with valid sqft values;

        print(self.df_combined_file.shape)

        new_df_with_sqft_value=self.df_combined_file[~self.df_combined_file.SQUARE_FEET.isna()]
        print('sqft with value')
        print(new_df_with_sqft_value.shape)

        new_df_with_sqft_na = self.df_combined_file[self.df_combined_file.SQUARE_FEET.isna()]
        print('sqft without value')
        print(new_df_with_sqft_na.shape)

        temp_df = new_df_with_sqft_value[['BEDS', 'BATHS','SQUARE_FEET']]

        corr = temp_df.corr()

        temp= self.df_combined_file[['BEDS', 'BATHS','SQUARE_FEET']]


        new_temp= self.df_combined_file[['BEDS', 'BATHS','SQUARE_FEET']]

        #new_temp['SQUARE_FEET'].fillna(temp.groupby(['BEDS', 'BATHS'])['SQUARE_FEET'].transform('mean'))
        self.df_combined_file['SQUARE_FEET']=temp.groupby(['BEDS', 'BATHS'])['SQUARE_FEET'].transform(lambda x: x.fillna(x.mean()))



        # Removing rows with sqft is na
        self.df_combined_file = self.df_combined_file[~self.df_combined_file.SQUARE_FEET.isna()]
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)

    # Method to remove the records with zip code length other than 5
    def clean_ZipCode(self):
        print(list(self.df_combined_file['ZIP_OR_POSTAL_CODE'].unique()))
        print((len(list(self.df_combined_file['ZIP_OR_POSTAL_CODE'].unique()))
               ))


        self.df_combined_file= self.df_combined_file[~(self.df_combined_file['ZIP_OR_POSTAL_CODE'].str.len() !=5)]
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)

        print(self.df_combined_file)

    # Method to create the csv file for cleaned data
    def create_csv(self):
        print("copying the dataframe to a new csv file")
        print(list(self.df_combined_file['ZIP_OR_POSTAL_CODE'].unique()))
        self.df_combined_file.to_csv("cleanedData.csv",index=False)

    # Method to clean the city feature of the dataset
    def clean_City(self):

        self.df_combined_file['CITY'] = self.df_combined_file['CITY'].str.upper()
        self.df_combined_file['CITY'] = self.df_combined_file['CITY'].dropna()

        self.df_combined_file['CITY numeric'] = self.df_combined_file['CITY'].apply(lambda r: self.findCharacterSum(r))

    # Method to find the character sum of city and convert the city to an equivalent numeric value
    def findCharacterSum(self, s):
        sum = 0
        if (type(s) != str):
            return 0

        for j in range(len(s)):
            sum += ord(s[j])

        return sum

def main():
    print("inside Main")
    print('path is ',path)

    obj = cleanData()
    obj.set_dir(path)
    obj.loadData("combined.csv")
    obj.identify_columns_with_missing_values()
    obj.uniqueValues()
    obj.corr_plots_price_lat_long()
    obj.data_price_yearbuilt()
    obj.data_price_age()

    obj.clean_PropertyType()
    obj.keep_necessary_columns()
    obj.remove_duplicates()
    obj.beds_Baths()
    obj.clean_sqft()
    obj.clean_ZipCode()
    obj.clean_City()
    obj.create_csv()

if __name__ == '__main__':
    main()
