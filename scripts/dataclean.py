import pandas as pd

def preprocess_for_lightgbm(dataset, columns_to_drop=None):
    # Splitting the DataFrame into object and numeric DataFrames
    current_data_object = dataset.select_dtypes(include=['object'])
    Current_data_Numericonly = dataset.select_dtypes(include=['number'])  # This includes int, float, etc.
    
    # Creating dummy variables for categorical data
    dummy_columns = pd.get_dummies(current_data_object, dtype=int)
    
    # Merging numeric data with dummy variables
    merged_data = pd.concat([Current_data_Numericonly, dummy_columns], axis=1)
    
    # Drop specified columns if any
    if columns_to_drop:
        merged_data.drop(columns=columns_to_drop, inplace=True)
    
    def clean_column_names(df):
        # Replace non-alphanumeric characters with underscore
        df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
        # Additional step to replace slashes with underscore, if not already covered
        df.columns = df.columns.str.replace('/', '_', regex=False)
        return df
    
    # Clean the column names of your DataFrame
    merged_data = clean_column_names(merged_data)
    
    # Optionally, save a sample of the merged data to a CSV file
    sample = merged_data.head(1)
    sample.to_csv('testmerge.csv')
    
    return merged_data

# Example usage:
# dataset = pd.read_csv('your_dataset.csv') # Load your dataset here
# columns_to_drop = ['SK_ID_CURR', 'AnotherColumnToDrop'] # Specify columns to drop here
# processed_data = preprocess_for_lightgbm(dataset, columns_to_drop)
# print(processed_data.dtypes)
# Now you can proceed with training your LightGBM model using `processed_data`
