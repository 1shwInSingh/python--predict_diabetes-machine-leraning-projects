import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # basic cleaning (just in case)
    df = df.drop_duplicates()
    return df