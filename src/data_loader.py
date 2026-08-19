import os
import pandas as pd
from typing import Optional

def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the student performance dataset from the provided path or default location.
    
    Parameters:
        file_path (str, optional): Path to the CSV file. If None, checks default paths.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if file_path is None:
        default_paths = [
            "data/student_performance_dataset.csv",
            "student_performance_dataset.csv",
            "../data/student_performance_dataset.csv",
            "../student_performance_dataset.csv"
        ]
        for path in default_paths:
            if os.path.exists(path):
                file_path = path
                break
                
    if file_path is None or not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    df = load_data()
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(df.head())
