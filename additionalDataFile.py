'''
Author : Latha Saradha
Purpose : This file is used to load the CSV files containing additional information around the city and locality of house
 and perform data pre-processing for the house features.
'''



import pandas as pd
import os
import math
import numpy as np
import time

import seaborn as sns
import pathlib

sns.set(font_scale=0.5)
import matplotlib.pyplot as plt


path= pathlib.Path().absolute()/"Data"
additional_data_path= pathlib.Path().absolute()/"Data"/"Additional_Data"

class additionalData:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Method to set the working directory
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

    # Method to load the house data
    def loadRedFinData(self, filename):
        print('Reading', filename)
        self.df_combined_file = (pd.read_csv(filename, index_col=False))

        print(self.df_combined_file.head())
        print(self.df_combined_file.shape)
        print(self.df_combined_file.columns)

    # Method to load the crime data
    def loadCrimeData(self, filename):
        print('Reading', filename)
        self.df_crime_file = (pd.read_csv(filename, index_col=False))

    # Method to load the school data
    def loadSchoolData(self, schoolrating, schoolLocation):
        print('Reading', schoolrating)
        self.df_school_rating = (pd.read_csv(schoolrating, encoding="utf-8", index_col=False))
        # Changing the column names with underscore
        self.df_school_rating.columns = [column.replace(" ", "_") for column in self.df_school_rating.columns]


        print('Reading', schoolLocation)
        self.df_school_location = (
            pd.read_csv(schoolLocation, encoding="ISO-8859-1", index_col=False, engine='python'))
        # Changing the column names with underscore
        self.df_school_location.columns = [column.replace(" ", "_") for column in self.df_school_location.columns]

    # Method to load the population data
    def load_population_data(self, population_file_name):
        print('Reading', population_file_name)
        self.df_population = (pd.read_csv(population_file_name, index_col=False))
        self.df_population.columns = [column.replace(" ", "_") for column in self.df_population.columns]

    # Method to load the hospital data
    def load_hospital_data(self, hospital_file_name):
        print('Reading', hospital_file_name)
        self.df_hospital = (pd.read_csv(hospital_file_name, index_col=False))
        self.df_hospital.columns = [column.replace(" ", "_") for column in self.df_hospital.columns]

    # Method to load the subway station data
    def load_subwayStations_data(self, subwayStations_file_name):
        print('Reading', subwayStations_file_name)
        self.df_subway = (pd.read_csv(subwayStations_file_name, index_col=False))
        self.df_subway.columns = [column.replace(" ", "_") for column in self.df_subway.columns]

    # Method to clean the unwantedcolumns in subway data
    def clean_subway_data(self):
        self.df_subway = self.df_subway[['Station_ID', 'Stop_Name', 'GTFS_Latitude', 'GTFS_Longitude']]

    # Method to find the number of subways available from each house record.
    def find_num_of_subways_for_each_house(self):
        print('Finding num of subway stations for each house ........ ')
        self.df_combined_file['Total_Num_of_Subways'] = 0
        self.df_combined_file['min_dist_station'] = 0

        # Find the number of subway stations for each house
        self.df_combined_file['Total_Num_of_Subways'] = self.df_combined_file[['LATITUDE', 'LONGITUDE']].apply(
            lambda row: self.haversine_np(row[1], row[0], self.df_subway['GTFS_Longitude'].values,
                                          self.df_subway['GTFS_Latitude'].values, 2.0), axis=1)


        for y in range(self.df_combined_file.shape[0]):
            lat1 = self.df_combined_file['LATITUDE'][y]
            long1 = self.df_combined_file['LONGITUDE'][y]
            min_dist = 100.0
            for x in range(self.df_subway.shape[0]):
                lat2 = self.df_subway['GTFS_Latitude'][x]
                long2 = self.df_subway['GTFS_Longitude'][x]
                dist = self.manhattan_distance(lat1, long1, lat2, long2)
                if (dist < min_dist):
                    min_dist = dist

            self.df_combined_file.loc[y, 'min_dist_station'] = min_dist

    # Method to clean the hospital data and remove unwanted columns
    def clean_hospital_data(self):
        # self.df_hospital['DBN'] = self.df_hospital['DBN'].str.strip()
        col_to_drop = ['Cross_Streets', 'Phone', 'Location_1', 'Postcode', 'Community_Board', 'Council_District',
                       'Census_Tract', 'BIN', 'BBL', 'NTA']
        self.df_hospital = self.df_hospital.drop(labels=col_to_drop, axis=1)

    # Method to find the number of health care facilities around house
    def find_num_of_healthcare_facilities(self):
        print('Finding num of healthcare facilities ')
        self.df_combined_file['Total_Num_ofHospitals'] = 0

        # Find the number of complaints for each house
        self.df_combined_file['Total_Num_ofHospitals'] = self.df_combined_file[['LATITUDE', 'LONGITUDE']].apply(
            lambda row: self.haversine_np(row[1], row[0], self.df_hospital['Longitude'].values,
                                          self.df_hospital['Latitude'].values, 5.0), axis=1)
        print(' After applying')

    # Method to clean the school data and remove un-necessary columns
    def clean_school_data(self):
        print("Merging ")
        # removing the extra white-space surrounding the text.
        self.df_school_rating['DBN'] = self.df_school_rating['DBN'].str.strip()
        self.df_school_location['ATS_SYSTEM_CODE'] = self.df_school_location['ATS_SYSTEM_CODE'].str.strip()

        # Merging 2 dataframes
        merged_inner = pd.merge(left=self.df_school_rating, how='inner', right=self.df_school_location, left_on='DBN',
                                right_on='ATS_SYSTEM_CODE')


        merged_inner = merged_inner[
            ['DBN', 'DISTRICT', 'SCHOOL', 'PRINCIPAL', 'PROGRESS_REPORT_TYPE', 'SCHOOL_LEVEL*', 'PEER_INDEX*',
             '2009-2010_OVERALL_GRADE', '2009-2010_OVERALL_SCORE', 'ATS_SYSTEM_CODE', 'Location_1']]



        # Differentiating the Latitude and Longitude Value
        merged_inner['LatLong'] = merged_inner['Location_1'].str.split('(').str[1]
        merged_inner['LatLong'] = merged_inner['LatLong'].str.strip()
        merged_inner['Latitude'] = merged_inner['LatLong'].str.split(',').str[0]
        merged_inner['Long'] = merged_inner['LatLong'].str.split(',').str[1]

        merged_inner['Longitude'] = merged_inner['Long'].str.replace(")", "")

        col_to_drop = ['LatLong', 'Long', 'DBN', 'DISTRICT', 'PRINCIPAL']
        merged_inner = merged_inner.drop(labels=col_to_drop, axis=1)

        merged_inner = merged_inner[~merged_inner['2009-2010_OVERALL_GRADE'].isna()]
        merged_inner = merged_inner.reset_index(drop=True)
        print(merged_inner.shape)
        self.df_combined_school_data = merged_inner

        self.df_combined_school_data.drop_duplicates(keep='first', inplace=True, ignore_index=True)
        self.df_combined_school_data = self.df_combined_school_data.reset_index(drop=True)
        print(self.df_combined_school_data.shape)

        # Converting the Latitude and Longitude to float datatype
        self.df_combined_school_data['Latitude'] = self.df_combined_school_data['Latitude'].astype(float)
        self.df_combined_school_data['Longitude'] = self.df_combined_school_data['Longitude'].astype(float)

        print("school data cleaning")
        print(self.df_combined_school_data.shape)

        self.df_combined_school_data = self.df_combined_school_data[~self.df_combined_school_data['Latitude'].isna()]

        self.df_combined_school_data = self.df_combined_school_data[~self.df_combined_school_data['Longitude'].isna()]

        self.df_combined_school_data = self.df_combined_school_data.reset_index(drop=True)

        print(self.df_combined_school_data.shape)

    # Method to find the school for each house record
    def find_school_for_each_house(self):

        print("Finding the school for each house.........")

        self.df_combined_school_data['Num_ofComplaints'] = self.df_combined_school_data['Num_ofComplaints'].astype(
            float)

        # Introducing new columns for School Level Count and Total Count
        self.df_combined_file['Level_A_SchoolCount'] = 0
        self.df_combined_file['Level_B_SchoolCount'] = 0
        self.df_combined_file['Level_C_SchoolCount'] = 0
        self.df_combined_file['Level_D_SchoolCount'] = 0
        self.df_combined_file['Level_F_SchoolCount'] = 0
        self.df_combined_file['Total_Number_of_Schools'] = 0
        self.df_combined_file['Num_Complaints_schools'] = 0

        for y in range(self.df_combined_file.shape[0]):
            lat1 = self.df_combined_file['LATITUDE'][y]
            long1 = self.df_combined_file['LONGITUDE'][y]
            min_dist = 3.0

            for x in range(self.df_combined_school_data.shape[0]):
                lat2 = self.df_combined_school_data['Latitude'][x]
                long2 = self.df_combined_school_data['Longitude'][x]
                dist = self.manhattan_distance(lat1, long1, lat2, long2)
                if dist <= min_dist:
                    temp = self.df_combined_school_data['2009-2010_OVERALL_GRADE'][x]
                    self.df_combined_file.loc[y, 'Total_Number_of_Schools'] += 1
                    count = self.df_combined_school_data['Num_ofComplaints'][x]
                    self.df_combined_file.loc[y, 'Num_Complaints_schools'] += count
                    if temp == 'A':
                        self.df_combined_file.loc[y, 'Level_A_SchoolCount'] += 1
                    elif temp == 'B':
                        self.df_combined_file.loc[y, 'Level_B_SchoolCount'] += 1
                    elif temp == 'C':
                        self.df_combined_file.loc[y, 'Level_C_SchoolCount'] += 1
                    elif temp == 'D':
                        self.df_combined_file.loc[y, 'Level_D_SchoolCount'] += 1
                    elif temp == 'F':
                        self.df_combined_file.loc[y, 'Level_F_SchoolCount'] += 1

        print(self.df_combined_file.shape)
        print(self.df_combined_file)

        print('Records with total number of schools =0')

        temp1 = self.df_combined_file.query('Total_Number_of_Schools==0')
        print(temp1)
        print(temp1.shape)

        X = self.df_combined_file['PRICE']
        Y = self.df_combined_file['Total_Number_of_Schools']

        scatter_plot_price_school = plt.scatter(x=X, y=Y)

        #plt.show()

    # Method to find the correlation between features
    def correlation_plot_combined_file(self):
        print(self.df_combined_file.columns)
        corr = self.df_combined_file.corr()
        corr = corr.round(2)
        print(corr)
        print(type(corr))

        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )

        #plt.show()

    # Method to clean the crime file data and remove unwanted columns
    def clean_crime_file(self):
        self.df_crime_file['Total_Crimes'] = self.df_crime_file['TOTAL_NON-SEVEN_MAJOR_FELONY_OFFENSES'] + \
                                             self.df_crime_file['TOTAL_SEVEN_MAJOR_FELONY_OFFENSES']

        col_to_drop = ['TOTAL_SEVEN_MAJOR_FELONY_OFFENSES', 'TOTAL_NON-SEVEN_MAJOR_FELONY_OFFENSES', 'Address',
                       'City_State', 'Address_Total', 'location']
        self.df_crime_file = self.df_crime_file.drop(labels=col_to_drop, axis=1)
        self.df_crime_file = self.df_crime_file.reset_index(drop=True)

        print(self.df_crime_file.head())
        print(self.df_crime_file.shape)
        print(self.df_crime_file.columns)

    # Method to find the precinct for each house
    def find_the_pct_for_each_house(self):

        self.df_combined_file['PCT'] = 0
        self.df_combined_file['Total_crimes'] = 0

        print("Finding the precinct for each house")

        for y in range(self.df_combined_file.shape[0]):
            lat1 = self.df_combined_file['LATITUDE'][y]
            long1 = self.df_combined_file['LONGITUDE'][y]
            min_dist = 100.0
            for x in range(self.df_crime_file.shape[0]):
                lat2 = self.df_crime_file['Latitude'][x]
                long2 = self.df_crime_file['Longitude'][x]
                dist = self.manhattan_distance(lat1, long1, lat2, long2)
                if (dist < min_dist):
                    min_dist = dist
                    pct = self.df_crime_file['PCT'][x]
                    total_crimes = self.df_crime_file['Total_Crimes'][x]
            self.df_combined_file.loc[y, 'PCT'] = pct
            self.df_combined_file.loc[y, 'Total_crimes'] = total_crimes

        print(self.df_combined_file)

        print(self.df_combined_file.query('PCT==0'))

    # Method to calculate the manhattan distance between two locations
    def manhattan_distance(self, lat1, lng1, lat2, lng2):

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
        total_distance = r * c * 0.621371
        return total_distance

    # Method to calculate the manhattan distance between two locations with array
    def manhattan_distance_parallel(self, lon1, lat1, lon2, lat2):

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        total_count = 0
        km = 6367 * c
        # print(type(km))
        # converting to miles
        km = km * 0.621371
        count = np.amin(km)
        return count

    # Method to calculate the circle distance between two locations
    def haversine_np(self, lon1, lat1, lon2, lat2, R):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        All args must be of equal length.
        """

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        total_count = 0
        km = 6367 * c

        # converting to miles
        km = km * 0.621371
        count = np.count_nonzero(km <= R)
        return count

    # Method to load the complaint data
    def loadComplaintData(self, filename):
        print('Reading', filename)
        self.df_complaints_data = (pd.read_csv( filename, index_col=False))

    # Method to clean the complaint files data
    def clean_complaints_file(self):
        self.df_complaints_data = self.df_complaints_data[
            ['CMPLNT_NUM', 'ADDR_PCT_CD', 'OFNS_DESC', 'Latitude', 'Longitude']]
        print(list(self.df_complaints_data['OFNS_DESC'].unique()))

    # Method to find the number of complaints for each house record
    def find_num_ofComplaints_for_each_house(self):
        print('Finding num of complaints........... ')
        self.df_combined_file['Total_Num_ofComplaints'] = 0

        # Find the number of complaints for each house
        self.df_combined_file['Total_Num_ofComplaints'] = self.df_combined_file[['LATITUDE', 'LONGITUDE']].apply(
            lambda row: self.haversine_np(row[1], row[0], self.df_complaints_data['Longitude'].values,
                                          self.df_complaints_data['Latitude'].values, 2.0), axis=1)
        print(' After applying')
        self.df_combined_file.merge(self.df_complaints_data, how='left', left_on='Total_Num_ofComplaints',
                                    right_index=True)
        print(self.df_combined_file.shape)
        print(self.df_combined_file)
        print(self.df_combined_file.columns)

        print("rows with 0 complaints")
        temp1 = self.df_combined_file.query('Total_Num_ofComplaints==0')
        print(temp1)

    # Method to clean the population data and remove un necessary columns
    def clean_population_data(self):
        # removing unwanted columns
        self.df_population = self.df_population[['Zip_Code', 'Population', 'People_/_Sq._Mile']]
        print("Printing Lines where values are N/A")

        temp = self.df_population.isnull()
        print(temp)

    # Method to find the population feature for each zip code data
    def find_population_per_zipcode(self):

        # Merging data from combined file and population file based on Zip code
        merged_inner = pd.merge(left=self.df_combined_file, how='inner', right=self.df_population,
                                left_on='ZIP_OR_POSTAL_CODE',
                                right_on='Zip_Code')

        col_to_drop = ['Zip_Code']
        merged_inner = merged_inner.drop(labels=col_to_drop, axis=1)
        # self.df_crime_file = self.df_crime_file.reset_index(drop=True)

        self.df_combined_file = merged_inner

        # changing the datatype of population and pop/sq mile

        self.df_combined_file["Population"] = self.df_combined_file["Population"].str.replace(",", "").astype(float)

        self.df_combined_file["People_/_Sq._Mile"] = self.df_combined_file["People_/_Sq._Mile"].str.replace(",", "")

        self.df_combined_file = self.df_combined_file.rename(columns={"People_/_Sq._Mile": "People/Sq_Mile"})

        # self.df_combined_file['People/Sq_Mile'] = self.df_combined_file['People/Sq_Mile'].str.strip()

        self.df_combined_file["People/Sq_Mile"] = self.df_combined_file["People/Sq_Mile"].astype(float)

    # Method to create an AdditionalDataAndHouseData.csv file
    def create_csv(self):
        print("copying the dataframe to a new csv file")
        self.df_combined_file.to_csv("AdditionalDataAndHouseData.csv", index=False)

    # Method to load the school safety data
    def load_school_safety_data(self, school_safety_filename):
        print('Reading', school_safety_filename)
        self.df_school_safety = (pd.read_csv(school_safety_filename, index_col=False))
        self.df_school_safety.columns = [column.replace(" ", "_") for column in self.df_school_safety.columns]

    # Method to clean the school safety data
    def clean_school_safety_data(self):
        self.df_school_safety = self.df_school_safety[
            ['School_Year', 'Location_Name', 'Address', 'Borough', 'Latitude', 'Longitude']]

        self.df_school_safety = self.df_school_safety[~self.df_school_safety['Latitude'].isna()]

        self.df_school_safety = self.df_school_safety[~self.df_school_safety['Longitude'].isna()]

        self.df_school_safety = self.df_school_safety.reset_index(drop=True)

        print(self.df_school_safety.shape)


        print(self.df_school_safety.columns[self.df_school_safety.isna().any()].tolist())

    # Method to find school safety for each School
    def find_school_safety_for_each_school(self):
        self.df_combined_school_data['Num_ofComplaints'] = 0

        print('Finding safety')

        self.df_combined_school_data['Num_ofComplaints'] = self.df_combined_school_data[
            ['Latitude', 'Longitude']].apply(
            lambda row: self.haversine_np(row[1], row[0], self.df_school_safety['Longitude'].values,
                                          self.df_school_safety['Latitude'].values, 0.1), axis=1)
        print('After applying')

        print(self.df_combined_school_data.shape)
        print(self.df_combined_school_data)
        print(self.df_combined_school_data.columns)

        print("rows with 0 complaints in schools")
        temp1 = self.df_combined_school_data.query('Num_ofComplaints==0')
        print(temp1)

    # Method to load the retail store data
    def load_retail_store_data(self, retail_store_filename):
        print('Reading', retail_store_filename)
        self.df_retail_store = (pd.read_csv(retail_store_filename, index_col=False))
        self.df_retail_store.columns = [column.replace(" ", "_") for column in self.df_retail_store.columns]

    # Method to clean the retail store data file
    def clean_retail_store_data(self):
        self.df_retail_store['LatLong'] = self.df_retail_store['Location'].str.split('(').str[1]
        self.df_retail_store['LatLong'] = self.df_retail_store['LatLong'].str.strip()
        self.df_retail_store['Latitude'] = self.df_retail_store['LatLong'].str.split(',').str[0]
        self.df_retail_store['Long'] = self.df_retail_store['LatLong'].str.split(',').str[1]

        self.df_retail_store['Longitude'] = self.df_retail_store['Long'].str.replace(")", "")

        self.df_retail_store = self.df_retail_store[
            ['Entity_Name', 'Address_Line_3', 'Zip_Code', 'Location', 'Latitude', 'Longitude']]

        self.df_retail_store = self.df_retail_store[~self.df_retail_store['Zip_Code'].isna()]
        self.df_retail_store = self.df_retail_store[~self.df_retail_store['Latitude'].isna()]
        self.df_retail_store = self.df_retail_store[~self.df_retail_store['Longitude'].isna()]

        self.df_retail_store['Latitude'] = self.df_retail_store['Latitude'].astype(float)
        self.df_retail_store['Longitude'] = self.df_retail_store['Longitude'].astype(float)
        self.df_retail_store = self.df_retail_store.reset_index(drop=True)

        print(self.df_retail_store.shape)
        print(self.df_retail_store)

    '''# Old version of the method
    def find_num_of_retail_stores_for_each_house(self):
        print("find_num_of_retail_stores_for_each_house")

        self.df_combined_file['Num_of_Retail_stores'] = 0
        self.df_combined_file['min_dist_retail_store'] = 0

        # Find the number of subway stations for each house
        self.df_combined_file['Num_of_Retail_stores'] = self.df_combined_file[['LATITUDE', 'LONGITUDE']].apply(
            lambda row: self.haversine_np(row[1], row[0], self.df_retail_store['Longitude'].values,
                                          self.df_retail_store['Latitude'].values, 1.0), axis=1)
        print('After applying')

        print(self.df_combined_file.shape)
        print(self.df_combined_file)

        print("Finding min distance retailstore")

        for y in range(self.df_combined_file.shape[0]):
            lat1 = self.df_combined_file['LATITUDE'][y]
            long1 = self.df_combined_file['LONGITUDE'][y]
            min_dist = 100.0
            print(y)
            for x in range(self.df_retail_store.shape[0]):
                lat2 = self.df_retail_store['Latitude'][x]
                long2 = self.df_retail_store['Longitude'][x]
                dist = self.manhattan_distance(lat1, long1, lat2, long2)
                if (dist < min_dist):
                    min_dist = dist

            self.df_combined_file.loc[y, 'min_dist_retail_store'] = min_dist

        print("Finding min distance retailstore")
        print(self.df_combined_file.shape)
        print(self.df_combined_file)

        print("Find the num of retailers using zipcode")
        self.df_combined_file['Num_of_Retail_stores_Zipcode'] = 0

        for y in range(self.df_combined_file.shape[0]):
            zip = self.df_combined_file['ZIP_OR_POSTAL_CODE'][y]
            print(y)
            for x in range(self.df_retail_store.shape[0]):
                zip_retail = self.df_retail_store['Zip_Code'][x]
                if (zip == zip_retail):
                    self.df_combined_file.loc[y, 'Num_of_Retail_stores_Zipcode'] += 1

        print(self.df_combined_file.shape)
        print(self.df_combined_file)'''

    # Method to find the number of reatil stores for each house
    def find_num_of_retail_stores_for_each_house_1(self):
        print("find_num_of_retail_stores_for_each_house")

        self.df_combined_file['Num_of_Retail_stores'] = 0
        self.df_combined_file['min_dist_retail_store'] = 0
        self.df_combined_file['Num_of_Retail_stores_Zipcode']=0

        # Find the number of subway stations for each house
        self.df_combined_file['Num_of_Retail_stores'] = self.df_combined_file[['LATITUDE', 'LONGITUDE']].apply(
            lambda row: self.haversine_np(row[1], row[0], self.df_retail_store['Longitude'].values,
                                          self.df_retail_store['Latitude'].values, 1.0), axis=1)


        print("Finding min distance retailstore")

        self.df_combined_file['min_dist_retail_store'] = self.df_combined_file[['LATITUDE', 'LONGITUDE']].apply(
            lambda row: self.manhattan_distance_parallel(row[1], row[0], self.df_retail_store['Longitude'].values,
                                                         self.df_retail_store['Latitude'].values), axis=1)

        self.df_retail_store['Num_of_Retail_stores_Zipcode']=0
        self.df_retail_store['Num_of_Retail_stores_Zipcode'] = self.df_retail_store.groupby('Zip_Code')[
            'Zip_Code'].transform('count')

        temp = self.df_retail_store.groupby(['Zip_Code']).count().reset_index()

        merged_inner = pd.merge(left=self.df_combined_file, how='left', right=temp, left_on='ZIP_OR_POSTAL_CODE',
                                right_on='Zip_Code')
        merged_inner.drop_duplicates(keep='first', inplace=False, ignore_index=True)

        print(merged_inner.columns)

        col_to_drop = ['Entity_Name', 'Address_Line_3', 'Zip_Code', 'Location', 'Latitude', 'Longitude','Num_of_Retail_stores_Zipcode_x']

        self.df_combined_file = merged_inner.drop(labels=col_to_drop, axis=1)

        print(self.df_combined_file.columns)
        self.df_combined_file = self.df_combined_file.rename(columns={"Num_of_Retail_stores_Zipcode_y": "Num_of_Retail_stores_Zipcode"})
        self.df_combined_file = self.df_combined_file.reset_index(drop=True)


        # changing the columns with nan value with 0
        self.df_combined_file['Num_of_Retail_stores_Zipcode'].fillna(0, inplace=True)


def main():
    print("inside Main")
    obj = additionalData()
    obj.set_dir(path)

    load_start_time = time.time()

    obj.loadRedFinData("cleanedData.csv")

    obj.set_dir(additional_data_path)

    # Crime related with complaints
    obj.loadComplaintData("NYPD_Complaint_Data_Current__Year_To_Date.csv")
    obj.clean_complaints_file()
    start_time = time.time()
    obj.find_num_ofComplaints_for_each_house()
    print("--- %s seconds ---" % (time.time() - start_time))

    # Crime related with precints
    obj.loadCrimeData("cleanedDataCrimeFile.csv")
    obj.clean_crime_file()
    obj.find_the_pct_for_each_house()

    # school related
    obj.loadSchoolData("School_Progress_Reports_-_All_Schools.csv", "School_Locations.csv")
    obj.clean_school_data()

    # School safety Data
    obj.load_school_safety_data("2010_-_2016_School_Safety_Report.csv")
    obj.clean_school_safety_data()
    obj.find_school_safety_for_each_school()
    obj.find_school_for_each_house()

    # population related
    obj.load_population_data("New_York_City_Population_By_Neighborhood_Tabulation_Areas.csv")
    obj.clean_population_data()
    obj.find_population_per_zipcode()

    # Hospital related data
    obj.load_hospital_data("NYC_Health___Hospitals_patient_care_locations_-_2011.csv")
    obj.clean_hospital_data()
    obj.find_num_of_healthcare_facilities()

    # Subway Station related Data
    obj.load_subwayStations_data("NYC_Subway_Station.csv")
    obj.clean_subway_data()
    obj.find_num_of_subways_for_each_house()


    # Retail Store Data
    obj.load_retail_store_data("Retail_Food_Stores.csv")
    obj.clean_retail_store_data()
    start_time = time.time()
    obj.find_num_of_retail_stores_for_each_house_1()
    print("--- %s seconds for retail stores ---" % (time.time() - start_time))

    # Finding correlation
    obj.correlation_plot_combined_file()

    # Create csv file
    obj.create_csv()
    print("--- %s seconds Total time ---" % (time.time() - load_start_time))


if __name__ == '__main__':
    main()
