import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/'


class cleanData:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    def __init__(self):
        df_combined_file = {}

    def loadData(self, filename):
        print('Reading', filename)
        self.df_combined_file = (pd.read_csv(path + filename,index_col=False))

        # commented to reduce prints
        '''
        print(self.df_combined_file.head())
        print(self.df_combined_file.shape)
        '''


    def identify_columns_with_missing_values(self):
        # commented to reduce prints
        '''
        print(list(self.df_combined_file.columns))



        #print((self.df_combined_file.columns[self.df_combined_file.isna().any()]).count)

        # Printing those column names where there is atleast one missing value
        print(dict(self.df_combined_file.isna().any()))

        print('Dimensions before removal are ',self.df_combined_file.shape)
        '''
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

        # commented to reduce prints
        '''
        print(self.df_combined_file.describe())

        print(self.df_combined_file.info())
        '''

    def corr_plots_price_lat_long(self):
        corr = self.df_combined_file[['PRICE','LATITUDE','LONGITUDE']].corr()
        print(corr)
        # commented to reduce prints
        '''
        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )

        plt.show()
        '''

    def data_price_yearbuilt(self):
        temp_df=self.df_combined_file[['PRICE','YEAR BUILT']]
        temp_df1=temp_df.dropna()

        # commented to reduce prints
        '''
        print(temp_df1.head())
        print(temp_df1.shape)
        '''
        corr = temp_df1.corr()
        print(corr)
        # commented to reduce prints
        '''
        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )

        plt.show()
        '''

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

        # commented to reduce prints
        '''
        print(temp_df1)
        print(temp_df1.shape)
        '''
        years=[1060,1055,1500,1025]
        self.df_combined_file.columns = [column.replace(" ", "_") for column in self.df_combined_file.columns]
        print(self.df_combined_file.query('YEAR_BUILT==1060.0'))

        # creating additional column to the dataframe
        self.df_combined_file['AGE']=2021-self.df_combined_file['YEAR_BUILT']

        # commented to reduce prints
        '''
        print(self.df_combined_file)
        print(self.df_combined_file.shape)
        
        scatter_plot_price_year=self.df_combined_file.plot.scatter(x='YEAR_BUILT',y='PRICE')
        plt.show()
        
        '''

        self.df_combined_file = self.df_combined_file[self.df_combined_file.YEAR_BUILT >= 1800.0]
        self.df_combined_file =self.df_combined_file.reset_index(drop=True)

        temp_df = self.df_combined_file[['PRICE', 'AGE']]
        temp_df1 = temp_df.dropna()

        # commented to reduce prints
        '''
        print(temp_df1)
        print(temp_df1.shape)
        '''
        corr = temp_df1.corr()
        print(corr)
        # commented to reduce prints
        '''
        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )

        plt.show()
        
        print("After removing age related columns")
        print(self.df_combined_file.describe())
        '''

    def clean_PropertyType(self):

        print()
        print(list(self.df_combined_file['PROPERTY_TYPE'].unique()))
        propertyTypes = ['Condo/Co-op', 'Single Family Residential', 'Multi-Family (2-4 Unit)', 'Townhouse','Multi-Family (5+ Unit)']
        self.df_combined_file = self.df_combined_file [self.df_combined_file.PROPERTY_TYPE.isin(propertyTypes)]
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)
        print(list(self.df_combined_file['PROPERTY_TYPE'].unique()))
        #commented to reduce prints
        '''
        print(self.df_combined_file)
        print(self.df_combined_file.shape)
        print(self.df_combined_file.describe())
        '''

        propertyTypes_Multi=['Multi-Family (2-4 Unit)', 'Multi-Family (5+ Unit)']
        new_datafram_MultiFamily=self.df_combined_file[self.df_combined_file.PROPERTY_TYPE.isin(propertyTypes_Multi)]

        # commented to reduce prints
        '''
        print(new_datafram_MultiFamily)
        print(new_datafram_MultiFamily.shape)
        print(list(new_datafram_MultiFamily['PROPERTY_TYPE'].unique()))
        print(new_datafram_MultiFamily.describe())
        '''

    def keep_necessary_columns(self):
        print(self.df_combined_file.shape)
        print(self.df_combined_file.columns)
        col_to_drop=['SALE_TYPE','SOLD_DATE','ADDRESS','DAYS_ON_MARKET','NEXT_OPEN_HOUSE_START_TIME','INTERESTED','FAVORITE','SOURCE','MLS#','SOURCE','LOT_SIZE','$/SQUARE_FEET','HOA/MONTH', 'STATUS', 'j','NEXT_OPEN_HOUSE_END_TIME','_']
        self.df_combined_file=self.df_combined_file.drop(labels=col_to_drop,axis=1)
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)

        # commented to reduce prints
        '''
        print(self.df_combined_file.columns)
        print(self.df_combined_file.shape)
        print(self.df_combined_file.describe())
        '''
        #PRICE, PROPERTY_TYPE, ZIP, BEDS, BATHS, SQUARE_FEET, LOCATION, YEAR_BUILT, URL, LATITUDE, LONGITUDE

    def remove_duplicates(self):
        print('Before removing duplicates')
        print(self.df_combined_file.shape)
        self.df_combined_file.drop_duplicates(keep='first',inplace=True,ignore_index=True)
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)
        print('After removing duplicates')
        print(self.df_combined_file.shape)

    def beds_Baths(self):

        #to find the BEDS calculation removing rows where there are no value of BEDS

        # commented to reduce prints
        print()
        '''
        print('After checking BEDS')
        print(list(self.df_combined_file['BEDS'].unique()))
        '''

        self.df_combined_file=self.df_combined_file [~self.df_combined_file.BEDS.isna()]
        print(self.df_combined_file)

        # Find calculations for Baths removing rows where there are no value of BATHS
        print('After checking BATHS')
        print(list(self.df_combined_file['BATHS'].unique()))
        self.df_combined_file = self.df_combined_file[~self.df_combined_file.BATHS.isna()]
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)
        print(self.df_combined_file.shape)
        # commented to reduce prints
        '''
        print(self.df_combined_file )
        print(self.df_combined_file .shape)
        print(list(self.df_combined_file ['BATHS'].unique()))
        '''
        print(list(self.df_combined_file['BEDS'].unique()))


        # commented to reduce prints
        '''
     #Histogram of Beds and Baths
     self.df_combined_file['BEDS'].plot.hist()
     self.df_combined_file['BATHS'].plot.hist()
     plt.show()
    
    
     # Printing those column names where there is atleast one missing value
     print(dict(self.df_combined_file.isna().any()))
    '''
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
        # commented to reduce prints
        '''
        print(corr)

        print('Unique num of beds in self.df_combined_file')
        print(list(self.df_combined_file['BEDS'].unique()))
        print('Unique num of beds in new_df_with_sqft_value')
        print(list(new_df_with_sqft_value['BEDS'].unique()))
        print('Unique num of beds in nnew_df_with_sqft_na')
        print(list(new_df_with_sqft_na['BEDS'].unique()))

        print('Unique num of BATHS in self.df_combined_file')
        print(list(self.df_combined_file['BATHS'].unique()))
        print('Unique num of BATHS in new_df_with_sqft_value')
        print(list(new_df_with_sqft_value['BATHS'].unique()))
        print('Unique num of BATHS in new_df_with_sqft_na')
        list1=list(new_df_with_sqft_na['BATHS'].unique())
        print(list1.sort(reverse=True))
        '''


        temp= self.df_combined_file[['BEDS', 'BATHS','SQUARE_FEET']]

        # commented to reduce prints
        '''
        print(temp.shape)
        print(temp)
        '''
        new_temp= self.df_combined_file[['BEDS', 'BATHS','SQUARE_FEET']]

        #new_temp['SQUARE_FEET'].fillna(temp.groupby(['BEDS', 'BATHS'])['SQUARE_FEET'].transform('mean'))
        self.df_combined_file['SQUARE_FEET']=temp.groupby(['BEDS', 'BATHS'])['SQUARE_FEET'].transform(lambda x: x.fillna(x.mean()))

        # commented to reduce prints
        '''
        print('After changing')

        print(self.df_combined_file.shape)
        print(self.df_combined_file)

        '''

        # Removing rows with sqft is na
        self.df_combined_file = self.df_combined_file[~self.df_combined_file.SQUARE_FEET.isna()]
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)

        # commented to reduce prints
        '''
        print('sqft without value')
        print(self.df_combined_file .shape)
        print(self.df_combined_file)
        '''


    def clean_ZipCode(self):
        print(list(self.df_combined_file['ZIP_OR_POSTAL_CODE'].unique()))
        print((len(list(self.df_combined_file['ZIP_OR_POSTAL_CODE'].unique()))
               ))


        #temp = self.df_combined_file[self.df_combined_file['ZIP_OR_POSTAL_CODE'].isin(options)]

        self.df_combined_file= self.df_combined_file[~(self.df_combined_file['ZIP_OR_POSTAL_CODE'].str.len() !=5)]
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)
        '''
        print('Zipcdes with no population data')
        options = ['11690', '10080', '10313', '11351','11380'                 ]
        temp = self.df_combined_file[self.df_combined_file['ZIP_OR_POSTAL_CODE'].isin(options)]
        print(temp)
        print(len(temp))
        '''
        print(self.df_combined_file)

    def create_csv(self):
        print("copying the dataframe to a new csv file")
        print(list(self.df_combined_file['ZIP_OR_POSTAL_CODE'].unique()))


        #self.df_combined_file.to_csv(path+"cleanedData.csv",index=False)

def main():
    print("inside Main")
    obj = cleanData()
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
    obj.create_csv()

if __name__ == '__main__':
    main()
