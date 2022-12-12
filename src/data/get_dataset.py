import pandas as pd

def get_dataset():
    return pd.read_csv("processed/stock_history_final_dataset.csv")