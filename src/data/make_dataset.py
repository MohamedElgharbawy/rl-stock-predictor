import pandas as pd
import glob
import os

def make_dataset():
    all_files = glob.glob(os.path.join("data/raw/", "*.csv"))

    df_from_each_file = [pd.read_csv(f) for f in all_files]
    for df, f in zip(df_from_each_file, all_files):
        df.insert(0, "Symbol", "")
        df["Symbol"] = os.path.basename(f).split(".")[0]

    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.to_csv("data/processed/final_dataset.csv", index=False)

if __name__ == "__main__":
    make_dataset()
