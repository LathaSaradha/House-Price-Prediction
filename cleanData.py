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
        print(self.df_combined_file.head())
        print(self.df_combined_file.shape)


    def identify_columns_with_missing_values(self):
        print(list(self.df_combined_file.columns))
        print(self.df_combined_file.isna)
        #print((self.df_combined_file.columns[self.df_combined_file.isna().any()]).count)
        # Printing those column names where there is atleast one missing value
        print(dict(self.df_combined_file.isna().any()))

        print('Dimensions before removal are ',self.df_combined_file.shape)

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
        print(self.df_combined_file.describe())

        print(self.df_combined_file.info())

    def corr_plots_price_lat_long(self):
        corr = self.df_combined_file[['PRICE','LATITUDE','LONGITUDE']].corr()
        print(corr)



        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )

        plt.show()

    def data_price_yearbuilt(self):
        temp_df=self.df_combined_file[['PRICE','YEAR BUILT']]
        temp_df1=temp_df.dropna()
        print(temp_df1.head())
        print(temp_df1.shape)

        corr = temp_df1.corr()
        print(corr)

        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )

        plt.show()

def main():
    print("inside Main")
    obj = cleanData()
    obj.loadData("combined.csv")
    obj.identify_columns_with_missing_values()
    obj.uniqueValues()
    obj.coorelation_plots()
    obj.data_price_yearbuilt()


if __name__ == '__main__':
    main()
