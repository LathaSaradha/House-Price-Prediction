'''
Author : Latha Saradha
Purpose : This file is a single point of file to call all the loading and
cleaning of data files and combine the house and external features.

'''
import pathlib
import os
import time

from additionalDataFile import additionalData
from cleanData import cleanData
from loadData import loadData

path= pathlib.Path().absolute()/"Data/"

additional_data_path= pathlib.Path().absolute()/"Data"/"Additional_Data"
class MainFile:

    def set_dir(self, path):
        try:
            os.chdir(path)
            print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")

        print(os.getcwd())

     # Method to clean the data
    def cleaningData(self):
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

    # Method to load the data
    def loadData(self):
        obj = loadData()
        obj.set_dir(path)
        txt = obj.get_files()
        concatenated_df = obj.load_file(txt)
        obj.create_csv(concatenated_df)

    # Method to load additional data
    def load_additional_Data(self):
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



def main():
    print("inside Main")
    print('path is ',path)
    obj=MainFile()

    obj.set_dir(path)

    # Loading the csv files
    obj.loadData()

    # Cleaning the house data
    obj.cleaningData()

    # Loading the additional data and combine with house data
    obj.load_additional_Data()


if __name__ == '__main__':
    main()
