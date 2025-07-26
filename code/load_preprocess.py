import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import os

def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target'] = iris.target
    return df

def preprocess_data(df):
    features = df.drop('target', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled['target'] = df['target']
    return df_scaled

def save_data(df_raw, df_processed):
    os.makedirs('../dataset/raw', exist_ok=True)
    os.makedirs('../dataset/processed', exist_ok=True)
    df_raw.to_csv('../dataset/raw/iris.csv', index=False)
    df_processed.to_csv('../dataset/processed/iris_scaled.csv', index=False)

if __name__ == "__main__":
    raw_df = load_data()
    processed_df = preprocess_data(raw_df)
    save_data(raw_df, processed_df)
