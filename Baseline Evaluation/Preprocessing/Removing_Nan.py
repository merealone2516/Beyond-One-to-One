import pandas as pd

# Function to clean dataset and remove all rows for any Pull_Label that has NaN values
def clean_dataset(file_path, save_path):

    df = pd.read_csv(file_path)

    print(f"\nRows with NaN values in {file_path}:")
    nan_rows = df[df.isnull().any(axis=1)]
    print(nan_rows)

    nan_values = df.isnull().sum()
    print(f"\nCount of NaN values in each column before deletion in {file_path}:")
    print(nan_values)

    unique_pull_labels_before = df['Pull_Label'].nunique()
    print(f"\nNumber of unique Pull_Labels before deletion in {file_path}: {unique_pull_labels_before}")

    labels_with_nan = df[df.isnull().any(axis=1)]['Pull_Label'].unique()

    df_cleaned = df[~df['Pull_Label'].isin(labels_with_nan)]

    nan_values_after = df_cleaned.isnull().sum()
    print(f"\nCount of NaN values in each column after deletion in {file_path}:")
    print(nan_values_after)

    unique_pull_labels_after = df_cleaned['Pull_Label'].nunique()
    print(f"\nNumber of unique Pull_Labels after deletion in {file_path}: {unique_pull_labels_after}")

    df_cleaned.to_csv(save_path, index=False)
    print(f"\nCleaned dataset saved to {save_path}")

# Clean both true and false datasets
clean_dataset('The path of the true files obtained after running the Python script Cleaning_and_preprocessing.py', 'path to save the files obtained')
clean_dataset('The path of the false files obtained after running the Python script Cleaning_and_preprocessing.py', 'path to save the files obtained')
