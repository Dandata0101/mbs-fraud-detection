import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_value_counts(data, column_name):
    """
    Plots horizontal bar chart and pie chart for the value counts of a specified column in the DataFrame.

    Parameters:
    - data: pandas.DataFrame, the DataFrame containing the data.
    - column_name: str, the name of the column to plot value counts for.
    """

    # Plotting setup
    plt.figure(figsize=(30, 12))

    # Horizontal Bar Chart for column value counts
    plt.subplot(1, 2, 1)
    value_counts = data[column_name].value_counts()
    bars = value_counts.plot(kind='barh', color=['#9e7edf', '#FFD700'])
    for index, value in enumerate(value_counts):
        # Shadow effect for text
        plt.text(value, index, str(value), va='center', ha='right', color='gray', fontsize=12, alpha=0.8, fontweight='bold')
        plt.text(value-1000, index, str(value), va='center', ha='right', color='black', fontsize=12, fontweight='bold')  # Actual text

    # Pie Chart for column value counts
    plt.subplot(1, 2, 2)
    colors = ['#9e7edf', '#FFD700']
    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%\n({v:d})'.format(p=pct, v=val)
        return my_format

    value_counts.plot(kind='pie', colors=colors, autopct=autopct_format(value_counts), startangle=140, shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold'})

    plt.ylabel('')  # Hide the column label on y-axis for the pie chart
    plt.title(f'{column_name} Variable Distribution')

    plt.tight_layout()  # Adjust layout to not overlap subplots
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

def distribution_graphs_objects(data, feature_columns, target_column, column_descriptions=None):
    """
    Plots distribution graphs for specified feature columns, grouped by a target column, and shows a table view of the data for each feature.

    Parameters:
    - data: pandas.DataFrame, the DataFrame containing the data.
    - feature_columns: list of str, the names of the feature columns to plot.
    - target_column: str, the name of the target column to group by.
    - column_descriptions: dict, (optional) a dictionary providing descriptions for columns.
    """
    for feature_column in feature_columns:
        print(f"\033[1m\033[1;3mDistribution Based on {feature_column}\033[0m")
        if column_descriptions and feature_column in column_descriptions:
            print(f'Description: {column_descriptions[feature_column]}\n')
        
        # Create a normalized value count converted to percentage
        target_group = round(data.groupby(target_column)[feature_column].value_counts(normalize=True, sort=False) * 100)
        
        # Calculate number of unique values in the target group to determine the number of colors needed
        cnt = int(target_group.groupby(level=0).count().max())
        
        all_colors = ['#F38181', '#FCE38A', '#EAFFD0', '#95E1D3', '#EEEEEE', '#00ADB5']
        colors = all_colors[:cnt]
        plt.figure(figsize=(30, 6))
        
        # Bar chart of distribution grouped by the target column
        plt.subplot(121)
        plt.title(f'{feature_column} Distribution grouped by {target_column}')
        ax = target_group.unstack().plot(kind='bar', stacked=True, color=colors)
        
        # Adding text labels on bars
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2, 
                    bar.get_y() + bar.get_height() / 2, 
                    f'{round(bar.get_height(), 1)}%', 
                    ha='center', 
                    va='bottom')

        # Pie chart of overall distribution
        plt.subplot(122)
        plt.title(f'{feature_column} distribution in Overall Records')
        (data[feature_column].value_counts(normalize=True) * 100).plot(kind='pie', autopct="%1.0f%%", colors=colors[:data[feature_column].nunique()])
        plt.ylabel('')

        plt.show()
        
        print("\033[1m\033[1;3mTable View\033[0m")
        print(target_group.unstack())
        print("\n\n")

# Example usage:
# Assuming `current_data` is your DataFrame, and you're interested in plotting distributions for a list of feature columns ['feature1', 'feature2', 'feature3', 'feature4'] grouped by 'TARGET'.
# features_list = ['feature1', 'feature2', 'feature3', 'feature4']
# column_descriptions = {'feature1': 'Description of feature1', ...}
# distribution_graphs(current_data, features_list, 'TARGET', column_descriptions)

def visualize_specific_distributions(data, features_list, segmentation_column=None, exclude_columns=None):
    """
    Visualizes the distribution of specified numeric data in the DataFrame, potentially segmented by a specified column,
    and excluding specified columns from the visualization.

    Parameters:
    - data: pandas.DataFrame, the DataFrame to process.
    - features_list: list of str, the specific columns to visualize.
    - segmentation_column: str, the column to use for segmenting the data in visualizations (optional).
    - exclude_columns: list of str, additional columns to exclude from visualization (optional).
    """
    # Clone the DataFrame to avoid modifying the original data
    data_processed = data.copy()
    
    # Ensure features_list does not include any columns to be excluded
    if exclude_columns is None:
        exclude_columns = []
    else:
        features_list = [col for col in features_list if col not in exclude_columns]
    if segmentation_column:
        exclude_columns.append(segmentation_column)
    
    # Focus on specified columns for visualization, excluding any non-numeric ones
    numeric_features_list = data_processed.select_dtypes(include=['float64', 'int64']).columns.intersection(features_list)
    
    print(f"Visualizing distributions for specified numeric columns: {numeric_features_list}.")

    # Visualize distributions for specified numeric columns
    for col in numeric_features_list:
        plt.figure(figsize=(18, 6))
        
        # Distribution plot
        plt.subplot(1, 2, 1)
        if segmentation_column:
            sns.histplot(data=data_processed, x=col, hue=segmentation_column, bins=10, palette='viridis', kde=True)
        else:
            sns.histplot(data=data_processed, x=col, bins=10, color='#222831', kde=True)
        plt.title(f'Distribution of {col}')

        # Box plot for overall data, segmented by the segmentation column if provided
        plt.subplot(1, 2, 2)
        if segmentation_column:
            sns.boxplot(x=segmentation_column, y=col, data=data_processed, palette='viridis')
            plt.title(f'{col} Distribution by {segmentation_column}')
        else:
            sns.boxplot(y=col, data=data_processed, color='#F38181')
            plt.title(f'Overall Boxplot of {col}')

        plt.tight_layout()
        plt.show()
