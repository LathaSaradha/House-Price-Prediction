import os

import pandas as pd
from geopy.extra.rate_limiter import RateLimiter

from geopy import Nominatim

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/Additional Data/'


class additionalData:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    def __init__(self):
        df_crime_file = {}

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


    def loadData(self, filename):
        print('Reading', filename)
        self.df_crime_file = (pd.read_csv(path + filename,index_col=False))

        print(self.df_crime_file)
        print(self.df_crime_file.shape)
        print(self.df_crime_file.columns)

    def find_latitude_long(self):
        self.df_crime_file.columns = [column.replace(" ", "_") for column in self.df_crime_file.columns]
        self.df_crime_file['Address_Total']=self.df_crime_file['Address'].astype(str)+' '+self.df_crime_file['City_State'].astype(str)

        print(self.df_crime_file.columns)
        print(self.df_crime_file)

        locator = Nominatim(user_agent="myGeocoder")
        # 1 - conveneint function to delay between geocoding calls
        geocode = RateLimiter(locator.geocode, min_delay_seconds=4)
        # 2- - create location column
        self.df_crime_file['location'] = self.df_crime_file['Address_Total'].apply(geocode)
        # 3 - create longitude, latitude and altitude from location column (returns tuple)
        self.df_crime_file['point'] = self.df_crime_file['location'].apply(lambda loc: tuple(loc.point) if loc else None)
        # 4 - split point column into latitude, longitude and altitude columns
        #self.df_crime_file[['latitude', 'longitude', 'altitude']] = pd.DataFrame(self.df_crime_file['point'].tolist(), index=self.df_crime_file.index)
        print(self.df_crime_file.columns)
        print(self.df_crime_file)



    def create_csv(self):
        print("copying the dataframe to a new csv file")

        self.df_crime_file.to_csv(path+"cleanedDataCrimeFile.csv",index=False)


def main():
    print("inside Main")
    obj = additionalData()
    obj.set_dir(path)
    obj.loadData('NYC_crime_total_data.csv')
    obj.find_latitude_long()
    obj.create_csv()



if __name__ == '__main__':
    main()