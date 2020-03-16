import os
import glob


import pandas as pd

path = 'C:/Users/Latha/Desktop/LATHA/Northeastern Illinois U/Masters Project/Data/'


class loadData:

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

    def get_files(self):
        txtfiles = []
        for file in glob.glob("*.csv"):
            txtfiles.append(file)
        return txtfiles

    def load_file(self,all_files):
        #all_files = glob.glob(os.path.join(path, "*.csv"))  # advisable to use os.path.join as this makes concatenation OS independent

        df_from_each_file = (pd.read_csv(f) for f in all_files)
        concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
        print(concatenated_df.shape)
        print(concatenated_df.head())
        return concatenated_df

    def create_csv(self,concatenated_df):
        print("copying the dataframe to a new csv file")
        concatenated_df.to_csv(path+"combined.csv")


def main():
    print("inside Main")
    obj = loadData()
    obj.set_dir(path)
    txt=obj.get_files()
    print(txt)
    concatenated_df=obj.load_file(txt)
    obj.create_csv(concatenated_df)



if __name__ == '__main__':
    main()



