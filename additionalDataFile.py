import pandas as pd
import os
import math
import numpy as np
import time

import seaborn as sns
import matplotlib.pyplot as plt

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/Additional Data/'


class additionalData:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    def __init__(self):
        df_crime_file = {}
        df_combined_file = {}
        df_school_rating={}
        df_school_location={}
        df_combined_school_data={}
        df_complaints_data={}
        df_population={}

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
        '''
        print(self.df_crime_file.head())
        print(self.df_crime_file.shape)
        print(self.df_crime_file.columns)
        '''

    def loadSchoolData(self,schoolrating,schoolLocation):
        print('Reading', schoolrating)
        self.df_school_rating = (pd.read_csv(path + schoolrating,encoding = "utf-8", index_col=False))
        # Changing the column names with underscore
        self.df_school_rating.columns = [column.replace(" ", "_") for column in self.df_school_rating.columns]

        # commented to reduce prints
        '''
        print(self.df_school_rating.head())
        print(self.df_school_rating.shape)
        print(self.df_school_rating.columns)
        '''
        print('Reading', schoolLocation)
        self.df_school_location = (pd.read_csv(path + schoolLocation, encoding = "ISO-8859-1",index_col=False,engine='python'))
        # Changing the column names with underscore
        self.df_school_location.columns = [column.replace(" ", "_") for column in self.df_school_location.columns]

        # commented to reduce prints
        '''
        print(self.df_school_location.head())
        print(self.df_school_location.shape)
        print(self.df_school_location.columns)
        '''
    def load_population_data(self,population_file_name):
        print('Reading', population_file_name)
        self.df_population = (pd.read_csv(path + population_file_name, index_col=False))
        self.df_population.columns = [column.replace(" ", "_") for column in self.df_population.columns]

        # commented to reduce prints
        '''
        print(self.df_population)
        print(self.df_population.shape)
        print(self.df_population.columns)
        '''

    def load_hospital_data(self, hospital_file_name):
        print('Reading', hospital_file_name)
        self.df_hospital = (pd.read_csv(path + hospital_file_name, index_col=False))
        self.df_hospital.columns = [column.replace(" ", "_") for column in self.df_hospital.columns]

        # commented to reduce prints
        '''
        print(self.df_hospital)
        print(self.df_hospital.shape)
        print(self.df_hospital.columns)
        '''

    def clean_hospital_data(self):
        #self.df_hospital['DBN'] = self.df_hospital['DBN'].str.strip()
        col_to_drop = ['Cross_Streets', 'Phone', 'Location_1', 'Postcode', 'Community_Board', 'Council_District', 'Census_Tract', 'BIN', 'BBL', 'NTA']
        self.df_hospital = self.df_hospital.drop(labels=col_to_drop, axis=1)
        # commented to reduce prints
        '''
        print(self.df_hospital.shape)
        print(self.df_hospital.columns)
        '''
    def find_num_of_healthcare_facilities(self):
        print('Finding num of healthcare facilities ')
        self.df_combined_file['Total_Num_ofHospitals'] = 0

        # Find the number of complaints for each house
        self.df_combined_file['Total_Num_ofHospitals'] = self.df_combined_file[['LATITUDE', 'LONGITUDE']].apply(
            lambda row: self.haversine_np(row[1], row[0], self.df_hospital['Longitude'].values,
                                          self.df_hospital['Latitude'].values, 5.0), axis=1)
        print(' After applying')
        print(self.df_combined_file.shape)

        print(self.df_combined_file.columns)

        # commented to reduce prints
        '''
            
        print(self.df_hospital.shape)
        print(self.df_hospital)
        print(self.df_hospital.columns)
        
        '''

        print("rows with 0 hospitals")
        temp1 = self.df_combined_file.query('Total_Num_ofHospitals==0')
        print(temp1)


    def clean_school_data(self):
        print("Merging ")
        # removing the extra white-space surrounding the text.
        self.df_school_rating['DBN'] = self.df_school_rating['DBN'].str.strip()
        self.df_school_location['ATS_SYSTEM_CODE'] = self.df_school_location['ATS_SYSTEM_CODE'].str.strip()

        # Merging 2 dataframes
        merged_inner = pd.merge(left=self.df_school_rating, how='inner', right=self.df_school_location, left_on='DBN',
                                right_on='ATS_SYSTEM_CODE')

        # commented to reduce prints
        '''
        print(merged_inner.head())
        print(merged_inner.shape)
        print(merged_inner.columns)
        '''
        merged_inner = merged_inner[['DBN', 'DISTRICT', 'SCHOOL', 'PRINCIPAL', 'PROGRESS_REPORT_TYPE', 'SCHOOL_LEVEL*', 'PEER_INDEX*',
             '2009-2010_OVERALL_GRADE', '2009-2010_OVERALL_SCORE', 'ATS_SYSTEM_CODE', 'Location_1']]

        # commented to reduce prints
        '''
        print(list(merged_inner['SCHOOL_LEVEL*'].unique()))
        print(list(merged_inner['2009-2010_OVERALL_GRADE'].unique()))
        '''


        # Differentiating the Latitude and Longitude Value
        merged_inner['LatLong']=merged_inner['Location_1'].str.split('(').str[1]
        merged_inner['LatLong'] = merged_inner['LatLong'].str.strip()
        merged_inner['Latitude']=merged_inner['LatLong'].str.split(',').str[0]
        merged_inner['Long']=merged_inner['LatLong'].str.split(',').str[1]

        merged_inner['Longitude']=merged_inner['Long'].str.replace(")","")


        col_to_drop = ['LatLong', 'Long','DBN','DISTRICT','PRINCIPAL']
        merged_inner = merged_inner.drop(labels=col_to_drop, axis=1)

        # commented to reduce prints
        '''
        print(merged_inner.head())
        print(merged_inner.shape)
        '''
        merged_inner = merged_inner[~merged_inner['2009-2010_OVERALL_GRADE'].isna()]
        merged_inner= merged_inner.reset_index(drop=True)
        print(merged_inner.shape)
        self.df_combined_school_data=merged_inner

    def find_school_for_each_house(self):

        print("Finding the school for each house")

        #Converting the Latitude and Longitude to float datatype
        self.df_combined_school_data['Latitude']=self.df_combined_school_data['Latitude'].astype(float)
        self.df_combined_school_data['Longitude'] = self.df_combined_school_data['Longitude'].astype(float)

       #Introducting new columns for School Level Count and Total Count
        self.df_combined_file['Level_A_SchoolCount'] = 0
        self.df_combined_file['Level_B_SchoolCount'] = 0
        self.df_combined_file['Level_C_SchoolCount'] = 0
        self.df_combined_file['Level_D_SchoolCount'] = 0
        self.df_combined_file['Level_F_SchoolCount'] = 0
        self.df_combined_file['Total_Number_of_Schools']=0



        for y in range(self.df_combined_file.shape[0]):
            lat1=self.df_combined_file['LATITUDE'][y]
            long1=self.df_combined_file['LONGITUDE'][y]
            min_dist=2.0
            for x in range(self.df_combined_school_data.shape[0]):
                lat2=self.df_combined_school_data['Latitude'][x]
                long2=self.df_combined_school_data['Longitude'][x]
                dist=self.manhattan_distance(lat1,long1,lat2,long2)
                if(dist<=min_dist):
                    temp=self.df_combined_school_data['2009-2010_OVERALL_GRADE'][x]

                    self.df_combined_file.loc[y, 'Total_Number_of_Schools'] += 1
                    if(temp=='A'):
                        self.df_combined_file.loc[y, 'Level_A_SchoolCount'] +=1
                    elif(temp=='B'):
                        self.df_combined_file.loc[y, 'Level_B_SchoolCount'] += 1
                    elif (temp == 'C'):
                        self.df_combined_file.loc[y, 'Level_C_SchoolCount'] += 1
                    elif (temp == 'D'):
                        self.df_combined_file.loc[y, 'Level_D_SchoolCount'] += 1
                    elif (temp == 'F'):
                        self.df_combined_file.loc[y, 'Level_F_SchoolCount'] += 1

        print(self.df_combined_file.shape)
        print(self.df_combined_file)

        temp1=self.df_combined_file.query('Total_Number_of_Schools==0')
        print(temp1)


    def correlation_plot_combined_file(self):
        print(self.df_combined_file.columns)
        corr = self.df_combined_file.corr()
        print(corr)
        # commented to reduce prints
        ax = sns.heatmap(
            corr,
            annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        plt.show()

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

        print(self.df_combined_file.query('PCT==0'))

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

    def haversine_np(self,lon1, lat1, lon2, lat2, R):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        All args must be of equal length.
        """
        #print("check")
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        total_count=0
        km = 6367 * c
        #print(type(km))
        count=np.count_nonzero(km<=R)
        return count

    def loadComplaintData(self, filename):
        print('Reading', filename)
        self.df_complaints_data = (pd.read_csv(path + filename, index_col=False))

        # commented to reduce prints

        print(self.df_complaints_data.head())
        print(self.df_complaints_data.shape)
        print(self.df_complaints_data.columns)

    def clean_complaints_file(self):
        self.df_complaints_data=self.df_complaints_data[['CMPLNT_NUM','ADDR_PCT_CD','OFNS_DESC','Latitude','Longitude']]
        print(list(self.df_complaints_data['OFNS_DESC'].unique()))


    def find_num_ofComplaints_for_each_house(self):
        print('Finding num of complaints ')
        self.df_combined_file['Total_Num_ofComplaints'] = 0

        # Find the number of complaints for each house
        self.df_combined_file['Total_Num_ofComplaints'] = self.df_combined_file[['LATITUDE', 'LONGITUDE']].apply(
            lambda row: self.haversine_np(row[1], row[0], self.df_complaints_data['Longitude'].values, self.df_complaints_data['Latitude'].values,1.0), axis=1)
        print(' After applying')
        self.df_combined_file.merge(self.df_complaints_data, how='left', left_on='Total_Num_ofComplaints', right_index=True)
        print(self.df_combined_file.shape)
        print(self.df_combined_file)
        print(self.df_combined_file.columns)

        print("rows with 0 complaints")
        temp1 = self.df_combined_file.query('Total_Num_ofComplaints==0')
        print(temp1)


    def test(self):
        print('Finding num of complaints ')
        #40.726787 -73.858789  1758
        # 40.694871 -73.924053 5087
        #40.913654 -73.897517 220
        lat1=40.913654
        long1= -73.897517
        total_num_of_complaints=0
        min_dist=1.0
        for x in range(self.df_complaints_data.shape[0]):
            lat2 = self.df_complaints_data['Latitude'][x]
            long2 = self.df_complaints_data['Longitude'][x]
            dist = self.manhattan_distance(lat1, long1, lat2, long2)
            if (dist <= min_dist):
                total_num_of_complaints += 1

        print(total_num_of_complaints)

    def clean_population_data(self):
        # removing unwanted columns
        self.df_population = self.df_population[['Zip_Code', 'Population', 'People_/_Sq._Mile']]
        print("Printing Lines where values are N/A")
      
        temp=self.df_population.isnull()
        print(temp)


    def find_population_per_zipcode(self):

        #Merging data from combined file and population file based on Zip code
        merged_inner = pd.merge(left=self.df_combined_file, how='inner', right=self.df_population, left_on='ZIP_OR_POSTAL_CODE',
                                right_on='Zip_Code')

        col_to_drop = ['Zip_Code']
        merged_inner = merged_inner.drop(labels=col_to_drop, axis=1)
        #self.df_crime_file = self.df_crime_file.reset_index(drop=True)

        self.df_combined_file=merged_inner

        #changing the datatype of population and pop/sq mile

        self.df_combined_file["Population"] = self.df_combined_file["Population"].str.replace(",", "").astype(float)

        self.df_combined_file["People_/_Sq._Mile"] = self.df_combined_file["People_/_Sq._Mile"].str.replace(",", "")

        self.df_combined_file=self.df_combined_file.rename(columns={"People_/_Sq._Mile": "People/Sq_Mile"})



        #self.df_combined_file['People/Sq_Mile'] = self.df_combined_file['People/Sq_Mile'].str.strip()

        self.df_combined_file["People/Sq_Mile"] = self.df_combined_file["People/Sq_Mile"].astype(float)

        # commented to reduce prints
        '''
        print(self.df_combined_file.head())
        
        print(self.df_combined_file.info())
        
        '''


def main():
    print("inside Main")
    obj = additionalData()
    obj.set_dir(path)
    obj.loadCrimeData("cleanedDataCrimeFile.csv")
    obj.loadRedFinData("cleanedData.csv")

    #Crime related with complaints
    obj.loadComplaintData("NYPD_Complaint_Data_Current__Year_To_Date.csv")
    obj.clean_complaints_file()
    start_time = time.time()
    obj.find_num_ofComplaints_for_each_house()

    print("--- %s seconds ---" % (time.time() - start_time))

    '''
    These can be removed.
    '''
    # Crime related with precints
    obj.clean_crime_file()
    obj.find_the_pct_for_each_house()
    obj.correlation_plot_combined_file()

    # school related
    obj.loadSchoolData("School_Progress_Reports_-_All_Schools.csv","School_Locations.csv")
    obj.clean_school_data()
    obj.find_school_for_each_house()

    #population related
    obj.load_population_data("New_York_City_Population_By_Neighborhood_Tabulation_Areas.csv")
    obj.clean_population_data()
    obj.find_population_per_zipcode()

    #Hospital related data
    obj.load_hospital_data("NYC_Health___Hospitals_patient_care_locations_-_2011.csv")
    obj.clean_hospital_data()
    obj.find_num_of_healthcare_facilities()

    #Finding correlation
    obj.correlation_plot_combined_file()

if __name__ == '__main__':
    main()
