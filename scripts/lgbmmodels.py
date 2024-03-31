import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_lightgbm_model(merged_data, target_column):
    # Fill missing values with 0
    merged_data = merged_data.fillna(0)
    
    # Assuming your DataFrame and target variable setup
    X = merged_data.drop(target_column, axis=1)
    y = merged_data[target_column]

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Parameters
    params = {
        'objective': 'binary', 
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
    }

    # Early stopping callback
    early_stopping_callback = lgb.early_stopping(stopping_rounds=10)

    # Training the model
    num_round = 100
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=num_round,
        valid_sets=[test_data],
        callbacks=[early_stopping_callback]
    )

    # Prediction
    y_pred_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred_proba]

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Accuracy: {accuracy}")
    
    return bst, accuracy,X_test

import shap
import matplotlib.pyplot as plt

def generate_shap_summary_plot(bst, X_test):
    """
    Generates a SHAP summary plot to show the impact of features on model predictions.

    Parameters:
    - bst: The trained LightGBM booster model.
    - X_test: The test dataset used to evaluate the model.
    """
    # Initialize a SHAP TreeExplainer with the trained LightGBM booster
    explainer = shap.TreeExplainer(bst)
    
    # Calculate SHAP values for the test dataset
    shap_values = explainer.shap_values(X_test)
    
    # Generate the SHAP summary plot
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    
    # Customize the plot
    plt.title("Impact of Features on Model Predictions with SHAP Values")
    plt.gcf().set_size_inches(10, 8)
    
    # Display the plot
    plt.show()

