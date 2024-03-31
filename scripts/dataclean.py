import pandas as pd
import os

def preprocess_for_lightgbm(dataset, columns_to_drop=None):
    if dataset is None:
        raise ValueError("The dataset provided is None. Please provide a valid Pandas DataFrame.")
    
    # Check if dataset is empty
    if dataset.empty:
        raise ValueError("The dataset provided is empty. Please provide a non-empty Pandas DataFrame.")
    
    # Ensure columns_to_drop is a list, even if empty or None
    columns_to_drop = columns_to_drop or []

    # Splitting the DataFrame into object and numeric DataFrames
    current_data_object = dataset.select_dtypes(include=['object'])
    current_data_numeric = dataset.select_dtypes(include=['number'])  # This includes int, float, etc.
    
    # Creating dummy variables for categorical data
    dummy_columns = pd.get_dummies(current_data_object, dtype=int)
    
    # Merging numeric data with dummy variables
    merged_data = pd.concat([current_data_numeric, dummy_columns], axis=1)
    
    # Drop specified columns, if any
    merged_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    def clean_column_names(df):
        # Replace non-alphanumeric characters with underscore
        df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
        return df
    
    # Clean the column names of your DataFrame
    merged_data = clean_column_names(merged_data)
    merged_data.fillna(0, inplace=True)

    # Optionally, save a sample of the merged data to a CSV file
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, '02-output')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create the directory if it does not exist
    
    sample_file_path = os.path.join(output_directory, 'testmerge.csv')
    sample = merged_data.head(1).T
    sample.to_csv(sample_file_path)
    
    print(f"Sample of the processed data saved to: {sample_file_path}")
    
    return merged_data

# Example usage (Assuming you have a DataFrame named `df5`):
# processed_data = preprocess_for_lightgbm(df5, columns_to_drop=['SK_ID_CURR'])
# print(processed_data.dtypes)
