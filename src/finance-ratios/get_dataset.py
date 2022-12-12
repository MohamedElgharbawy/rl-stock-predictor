import pandas as pd

def get_dataset():
    return pd.read_csv("../finance-ratios/processed/final_dataset.csv")