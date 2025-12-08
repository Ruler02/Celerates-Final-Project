import pickle
import pandas as pd

def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

def load_dataset():
    df = pd.read_csv("data\BreastCancer123.csv")
    return df.drop(columns=["diagnosis", "Unnamed: 32", "id"])
