import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

def evaluate_models_and_visualize_smote(df, target_variable_name, model_choices, columns_to_drop=None, top_n_features=10):
    if columns_to_drop is not None:
        df = df.drop(columns=columns_to_drop)
    
    x = df.drop(target_variable_name, axis=1)
    y = df[target_variable_name]
    
    # Apply SMOTE
    smote = SMOTE(random_state=0)
    X_resampled, y_resampled = smote.fit_resample(x, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=0)
    
    # Define available models within the function
    available_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(kernel='linear'),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Gaussian NB": GaussianNB()
    }
    
    # Ensure model_choices is a list to simplify processing
    if isinstance(model_choices, str):
        model_choices = [model_choices]
    
    for model_name in model_choices:
        model = available_models.get(model_name, None)
        
        if model is None:
            print(f"Model '{model_name}' not found. Skipping.")
            continue
        
        print(f"Evaluating: {model_name}")
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        
        # Print classification report
        print(f"{model_name} Classification Report:\n", classification_report(y_test, predicted))
        
        # Plot and display the confusion matrix
        cm = confusion_matrix(y_test, predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.show()
        
        # Plot feature importances for models that have this attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n_features:]
            plt.figure(figsize=(10, 8))
            bars = plt.barh(range(top_n_features), importances[indices], align='center', color='skyblue')
            plt.yticks(range(top_n_features), [x.columns[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.title(f'Top {top_n_features} Feature Importances for {model_name}')

            # Add the feature importances values on the bars
            for bar in bars:
                plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                         f'{bar.get_width():.3f}', 
                         va='center', ha='left', fontsize=8)
            plt.tight_layout()
            plt.show()

# Example usage:
# evaluate_models_and_visualize_smote(df, 'Target', ['Random Forest'], columns_to_drop=['Column1', 'Column2'], top_n_features=20)
