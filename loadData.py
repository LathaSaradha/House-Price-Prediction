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

    def set_dir(self,path):
        try:
            os.chdir(path)
            print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")

        print(os.getcwd())


    def get_files(self):
        txtfiles = []
        for file in glob.glob("*.csv"):
            txtfiles.append(file)
        return txtfiles

    def load_file(self,all_files):
        df_from_each_file = (pd.read_csv(f,index_col=False) for f in all_files)
        concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
        print(concatenated_df.shape)
        print(concatenated_df.head())
        return concatenated_df

    def create_csv(self,concatenated_df):
        print("copying the dataframe to a new csv file")
        concatenated_df.to_csv("combined.csv",index=False)


def main():
    print("inside Main")
    print('path is ',path)
    obj = loadData()
    obj.set_dir(path)
    txt=obj.get_files()
    print(txt)
    print()
    concatenated_df=obj.load_file(txt)
    obj.create_csv(concatenated_df)



if __name__ == '__main__':
    main()



