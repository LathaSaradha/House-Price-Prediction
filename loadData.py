'''
Author : Latha Saradha
Purpose : This file is used to load all the CSV files which contain the data for the house
 and create a combined csv file for further calculation.
'''


import os
import glob
import pathlib



import pandas as pd

path= pathlib.Path()/'Data'

class loadData:

    #This method is used to set the working directory
    def set_dir(self,path):
        try:
            os.chdir(path)
            print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")

        print(os.getcwd())

    # This method is to get all the file names with ".csv" extension. These csv files contain the house data
    def get_files(self):
        txtfiles = []
        for file in glob.glob("*.csv"):
            txtfiles.append(file)
        return txtfiles

    # This method is to load all the files stored in the txtfiles array of names of csv files
    def load_file(self,all_files):
        df_from_each_file = (pd.read_csv(f,index_col=False) for f in all_files)

        # The files are concatenated to a single dataframe
        concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
        print(concatenated_df.shape)
        print(concatenated_df.head())
        return concatenated_df

    # The concatenated dataframe is converted to a new csv file to use for further exploration
    def create_csv(self,concatenated_df):
        print("copying the dataframe to a new csv file")
        concatenated_df.to_csv("combined.csv",index=False)

def main():
    print("inside Main")
    print('path is ',path)
    obj = loadData()
    obj.set_dir(path)
    txt=obj.get_files()
    print('Files are ' ,txt)

    concatenated_df=obj.load_file(txt)
    obj.create_csv(concatenated_df)

if __name__ == '__main__':
    main()



