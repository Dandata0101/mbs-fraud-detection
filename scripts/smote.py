import pandas as pd
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

def evaluate_models_and_visualize_smote(df, target_variable_name, model_choices, columns_to_drop=None):
    if columns_to_drop is not None:
        df = df.drop(columns=columns_to_drop)
    
    x = df.drop(target_variable_name, axis=1)
    y = df[target_variable_name]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    
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

# Example usage:
# evaluate_models_and_visualize_with_confusion_matrix(df, 'Target', ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree', 'KNN', 'Gaussian NB'], ['Column1', 'Column2'])
