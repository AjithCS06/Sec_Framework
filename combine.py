import os
import pandas as pd

def combine_data():
    # Initialize an empty DataFrame for merging the datasets
    merged_data = pd.DataFrame()

    folder_path = "enc_app/static/Uploaded_Data"
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Read and merge each dataset
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        dataset = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, dataset])

    folder_path = 'enc_app/static/Merged/'
    os.makedirs(folder_path, exist_ok=True)

    # Save the merged dataset as a CSV file
    merged_filename = folder_path+"merged_dataset.csv"
    merged_data.to_csv(merged_filename, index=False)

    df=pd.read_csv(folder_path+"merged_dataset.csv")
    return df