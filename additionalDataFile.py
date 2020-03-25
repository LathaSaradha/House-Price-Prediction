import pandas as pd
import os
import math
import numpy as np

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/Additional Data/'


class additionalData:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    def __init__(self):
        df_crime_file = {}
        df_combined_file = {}

    def set_dir(self,path):
        try:
            os.chdir(path)
            print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")

        print(os.getcwd())
        if os.path.exists(path) :
            # Change the current working Directory
            os.chdir(path)
        else:
            print("Can't change the Current Working Directory")


    def loadRedFinData(self, filename):
        print('Reading', filename)
        self.df_combined_file = (pd.read_csv(path + filename,index_col=False))

        print(self.df_combined_file.head())
        print(self.df_combined_file.shape)
        print(self.df_combined_file.columns)


    def loadCrimeData(self, filename):
        print('Reading', filename)
        self.df_crime_file = (pd.read_csv(path + filename,index_col=False))

        # commented to reduce prints

        print(self.df_crime_file.head())
        print(self.df_crime_file.shape)
        print(self.df_crime_file.columns)

    def clean_crime_file(self):
        self.df_crime_file['Total_Crimes']=self.df_crime_file['TOTAL_NON-SEVEN_MAJOR_FELONY_OFFENSES']+self.df_crime_file['TOTAL_SEVEN_MAJOR_FELONY_OFFENSES']

        col_to_drop=['TOTAL_SEVEN_MAJOR_FELONY_OFFENSES','TOTAL_NON-SEVEN_MAJOR_FELONY_OFFENSES','Address','City_State','Address_Total','location']
        self.df_crime_file = self.df_crime_file.drop(labels=col_to_drop, axis=1)
        self.df_crime_file = self.df_crime_file.reset_index(drop=True)

        print(self.df_crime_file.head())
        print(self.df_crime_file.shape)
        print(self.df_crime_file.columns)

    def find_the_pct_for_each_house(self):

        self.df_combined_file['PCT']=0
        self.df_combined_file['Total_crimes']=0

        print("Finding the precinct for each house")


        for y in range(self.df_combined_file.shape[0]):
            lat1=self.df_combined_file['LATITUDE'][y]
            long1=self.df_combined_file['LONGITUDE'][y]
            min_dist=100.0
            for x in range(self.df_crime_file.shape[0]):
                lat2=self.df_crime_file['Latitude'][x]
                long2=self.df_crime_file['Longitude'][x]
                dist=self.manhattan_distance(lat1,long1,lat2,long2)
                if(dist<min_dist):
                    min_dist=dist
                    pct=self.df_crime_file['PCT'][x]
                    total_crimes=self.df_crime_file['Total_Crimes'][x]
            self.df_combined_file.loc[y,'PCT'] =pct
            self.df_combined_file.loc[y,'Total_crimes']=total_crimes


        print(self.df_combined_file)





    def manhattan_distance(self,lat1, lng1, lat2, lng2):

        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)
        lon1 = math.radians(lng1)
        lon2 = math.radians(lng2)
        d_lon = lon2 - lon1
        d_lat = lat2 - lat1

        # Radius of the earth
        r = 6373.0
        a = (np.sin(d_lat / 2.0)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(d_lon / 2.0)) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        total_distance = r * c
        return total_distance


def main():
    print("inside Main")
    obj = additionalData()
    obj.set_dir(path)
    obj.loadCrimeData("cleanedDataCrimeFile.csv")
    obj.loadRedFinData("cleanedData.csv")
    obj.clean_crime_file()
    obj.find_the_pct_for_each_house()

if __name__ == '__main__':
    main()
