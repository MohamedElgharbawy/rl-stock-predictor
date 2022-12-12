import pandas as pd
import glob
import os

def make_dataset():
    all_files = glob.glob(os.path.join("finance-ratios/raw/", "*.csv"))

    df_from_each_file = [pd.read_csv(f).set_index('period_end_date').T for f in all_files]
    for df, f in zip(df_from_each_file, all_files):
        df.insert(0, "Symbol", "")
        df["Symbol"] = os.path.basename(f).split(".")[0]
        df['period_end_date'] = df.index
        # print(df)

    concatenated_df = pd.concat(df_from_each_file)
    concatenated_df = concatenated_df[concatenated_df.revenue.notnull()]
    max_number_of_nas_cols = 100
    max_number_of_nas_rows = 10
    concatenated_df = concatenated_df.loc[:, (concatenated_df.isnull().sum(axis=0) <= max_number_of_nas_cols)]
    
    # print(concatenated_df.isnull().sum(axis=0).tolist())  # to get a list instead of an Index object
    concatenated_df = concatenated_df.loc[(concatenated_df.isnull().sum(axis=1) <= max_number_of_nas_rows), :]
    
    # comment this out if you want to interpolate instead of removing.
    concatenated_df = concatenated_df.loc[:, concatenated_df.notna().all()]
    
    # print(concatenated_df)  
    concatenated_df.to_csv("finance-ratios/processed/final_dataset.csv", index=False)

if __name__ == "__main__":
    make_dataset()
