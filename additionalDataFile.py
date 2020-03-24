import pandas as pd
import os

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/Additional Data/'


class additionalData:

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

        # commented to reduce prints

        print(self.df_crime_file.head())
        print(self.df_crime_file.shape)
        print(self.df_crime_file.columns)



def main():
    print("inside Main")
    obj = additionalData()
    obj.set_dir(path)
    obj.loadData("cleanedDataCrimeFile.csv")

if __name__ == '__main__':
    main()
