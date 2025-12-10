import pickle
import pandas as pd
from pathlib import Path

def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

def load_dataset():
    # Path otomatis menyesuaikan Windows/Linux
    file_path = Path(__file__).parent.parent / "data" / "BreastCancer123.csv"

    # Cek file ada atau tidak → lebih mudah debug
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    return df.drop(columns=["diagnosis", "Unnamed: 32", "id"])
