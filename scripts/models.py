import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import shap
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def logistic_regression_model(df, target_column, drop_columns=None, add_constant=True, return_type='model'):
    """
    Fits a logistic regression model to the given DataFrame. Depending on the return_type parameter,
    it prints McFadden's R^2, prints the full model summary, or returns the model object itself.

    Parameters:
    - df: DataFrame containing the data.
    - target_column: Name of the column in `df` that contains the target variable.
    - drop_columns: (Optional) List of column names in `df` to be dropped. Can include ID columns or any others not needed for the model.
    - add_constant: (Optional) Whether to add a constant (intercept) to the model. Default is True.
    - return_type: (Optional) Specifies the type of output: 'R2' for McFadden's R^2 and prints it, 'summary' to print the model summary, and 'model' to return the model object itself without printing. Default is 'model'.

    Returns:
    - McFadden's R^2 if return_type is 'R2', nothing if 'summary' (since it prints the summary), or the logistic regression model object if 'model'.
    """
    # Preparing the features and target variable by dropping specified columns
    X = df.drop(columns=[target_column] + (drop_columns if drop_columns is not None else []))
    Y = df[target_column]

    # Optionally adding a constant to the model (intercept)
    if add_constant:
        X = sm.add_constant(X)

    # Fit the model
    model = sm.Logit(Y, X).fit(disp=0) # `disp=0` suppresses the output during the fitting process

    if return_type == 'R2':
        # Calculate and print McFadden's R^2
        llf = model.llf  # Log-likelihood of the model
        llnull = model.llnull  # Log-likelihood of the null model
        R2_McFadden = 1 - (llf / llnull)
        print(f"Logistic R^2: {R2_McFadden}")
        return R2_McFadden
    elif return_type == 'summary':
        # Print the model summary
        print(model.summary())
    else:
        # Return the model object itself
        return model

def train_and_evaluate_decision_tree(df, target_column, drop_columns, test_size=0.3, random_state=42, return_accuracy_only=False, top_n_features=20):
    """
    Trains and evaluates a Decision Tree Classifier and visualizes the top N feature importances with values and the decision tree itself.
    """
    # Prepare the data
    columns_to_drop = drop_columns + [target_column]
    X = df.drop(columns=columns_to_drop)
    Y = df[target_column]

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Training the model
    dt_model = DecisionTreeClassifier(random_state=random_state)
    dt_model.fit(X_train, y_train)

    # Making predictions
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if return_accuracy_only:
        print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    else:
        # Print classification report
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Plotting the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

        # Calculate and plot feature importances
        feature_importances = dt_model.feature_importances_
        indices = np.argsort(feature_importances)[-top_n_features:]
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(top_n_features), feature_importances[indices], align='center', color='skyblue')
        plt.yticks(range(top_n_features), [X.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title('Top 20 Feature Importances')

        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                     f'{bar.get_width():.3f}', 
                     va='center', ha='left', fontsize=8)
        plt.tight_layout()
        plt.show()

        # Plot the decision tree
        plt.figure(figsize=(20, 10))
        plot_tree(dt_model, feature_names=X.columns, class_names=[str(cls) for cls in dt_model.classes_], filled=True, impurity=True, max_depth=3, fontsize=10)
        plt.title('Decision Tree')
        plt.show()

def train_and_evaluate_random_forest(df, target_column, drop_columns, test_size=0.3, random_state=42, return_accuracy_only=False, top_n_features=20):
    """
    Trains and evaluates a Random Forest Classifier and visualizes the top N feature importances with values.
    """
    # Prepare the data
    columns_to_drop = drop_columns + [target_column]
    X = df.drop(columns=columns_to_drop)
    Y = df[target_column]

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Training the model
    rf_model = RandomForestClassifier(random_state=random_state)
    rf_model.fit(X_train, y_train)

    # Making predictions
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if return_accuracy_only:
        print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    # Print classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Plotting the confusion matrix using ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Feature Importances
    feature_importances = rf_model.feature_importances_
    indices = np.argsort(feature_importances)[-top_n_features:]
    
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(top_n_features), feature_importances[indices], align='center', color='skyblue')
    plt.yticks(range(top_n_features), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Top 20 Feature Importances')

    # Add the feature importances values on the bars
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.3f}', 
                 va='center', ha='left', fontsize=8)
    plt.tight_layout()
    plt.show()

    # Plot one of the trees from the random forest
    plt.figure(figsize=(20, 10))
    tree_index = 0  # Choosing the first tree as an example
    plot_tree(rf_model.estimators_[tree_index], feature_names=X.columns, class_names=[str(cls) for cls in rf_model.classes_], filled=True, impurity=True, max_depth=3, fontsize=10)
    plt.title('Decision Tree from the Random Forest')
    plt.show()




def train_and_evaluate_knn(df, target_column, drop_columns, test_size=0.3, random_state=42, n_neighbors=5, return_accuracy_only=False):
    """
    Trains and evaluates a K-Nearest Neighbors Classifier and provides a confusion matrix.
    
    Parameters:
    - df: DataFrame containing the data.
    - target_column: string, the name of the target variable column.
    - drop_columns: list, column names to be dropped from df.
    - test_size: float, proportion of the dataset to include in the test split.
    - random_state: int, controls the shuffling applied to the data before applying the split.
    - n_neighbors: int, number of neighbors to use for KNN.
    - return_accuracy_only: bool, if True, prints and returns only the model's accuracy.
    
    Outputs:
    - If return_accuracy_only is True, prints and returns the model's accuracy.
    - Otherwise, prints the classification report and confusion matrix.
    """
    # Prepare the data
    columns_to_drop = drop_columns + [target_column]
    X = df.drop(columns=columns_to_drop)
    Y = df[target_column]

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Training the model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)

    # Making predictions
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if return_accuracy_only:
        print(f"KNN Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    else:
        # Print classification report
        report = classification_report(y_test, y_pred)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
def train_and_evaluate_gaussian_nb(df, target_column, drop_columns, test_size=0.3, random_state=42, return_accuracy_only=False):
    """
    Trains and evaluates a Gaussian Naive Bayes Classifier and visualizes the confusion matrix.
    
    """
    # Prepare the data
    columns_to_drop = drop_columns + [target_column]
    X = df.drop(columns=columns_to_drop)
    Y = df[target_column]

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Training the model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Making predictions
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if return_accuracy_only:
        print(f"Gaussian Naive Bayes Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    else:
        # Print classification report
        report = classification_report(y_test, y_pred)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()