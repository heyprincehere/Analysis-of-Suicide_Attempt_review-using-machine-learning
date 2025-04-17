import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Preprocessing function to handle NaN and encoding categorical variables
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    # Drop unnecessary columns
    X = data.drop(columns=['generation'])
    y = data['generation']

    # Handle columns that have commas in the numbers (e.g., 'gdp_for_year ($)')
    if 'gdp_for_year ($)' in X.columns:
        X['gdp_for_year ($)'] = X['gdp_for_year ($)'].replace({',': ''}, regex=True).astype(float)

    # Encode categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Impute numeric data with mean strategy
    numeric_imputer = SimpleImputer(strategy='mean')
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

    # Impute categorical data with the most frequent value before encoding
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

    # Label encode categorical variables after imputation
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        if col in X.columns:
            X[col] = label_encoder.fit_transform(X[col])

    return X,y

# Load the dataset
def load_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    return data


def trmr(X_train,y_train):
    model=RandomForestClassifier()
    model.fit(X_train,y_train)
    print("traine completed")
    return model

# Model evaluation function
def evaluate_models(models, X_test, y_test):
    #accuracies = {}
  
    y_pred = models.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'binary' for binary classification
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)


    return accuracy,precision,recall,y_pred

# Main execution
def main():
    # Path to your local CSV file
    file_path = r'C:\Users\Prince\OneDrive\Desktop\Analysis of Suicide_Attempt_review  using machine learning\CODE\Suicide_Attempt_review\media\train.csv'
    
    # Load and preprocess data
    data = load_data(file_path)
    print("load is completed")
    X, y = preprocess_data(data)
    print(X,y)
    print('processing_completed')
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('split completed')
    # Train the models
    models = trmr(X_train, y_train)
    print('training completed')
    # Evaluate the models
    print(X_test)
    accuracies,precison,recall,y_pred = evaluate_models(models, X_test, y_test)
    #pred = models.predict(X_test)
    print(y_pred)

    #conf_matrix = confusion_matrix(y_test, pred)

    """plt.figure(figsize=(7,5))
    sns.heatmap(data.corr(), annot=True, cmap='Oranges')
    plt.show()"""
    # Extract values from the confusion matrix

   # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=models.classes_, yticklabels=models.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Corelation Map')
    plt.show()

    print("Corelation Matrix:\n", cm)



    

    plt.figure(figsize=(9,5))
    sns.barplot(x=data['generation'], y=data['suicides_no'])
    plt.xlabel('Generation')
    plt.ylabel('Suicide Count')
    plt.title('Generation - Suicide Count Bar Plot')
    plt.show()


    
    return accuracies,precison,recall


# Placeholder for future prediction functionality


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prediction_value(message):
    file_path = r'C:\Users\Prince\OneDrive\Desktop\Analysis of Suicide_Attempt_review  using machine learning\CODE\Suicide_Attempt_review\media\train.csv'
    

    data = load_data(file_path)
    print('Data loaded successfully')

    # Prepare input DataFrame from the message
    input_df = pd.DataFrame([message])  # Wrap message in a list to create a DataFrame
    print("Input DataFrame:")
    print(input_df)

    # Define required columns and features
    required_columns = ['year', 'country', 'age', 'generation']
    for column in required_columns:
        if column not in data.columns:
            raise KeyError(f"Missing required column in dataset: {column}")

    features = ['year', 'country', 'age']  # Features used for prediction
    target = 'generation'  # The target variable

    # Handle categorical variables by encoding them
    country_encoder = LabelEncoder()
    age_encoder = LabelEncoder()

    # Fit label encoders on the entire dataset
    data['country'] = country_encoder.fit_transform(data['country'])
    data['age'] = age_encoder.fit_transform(data['age'])

    # Encode the input DataFrame (message)
    input_df['country'] = country_encoder.transform(input_df['country'])
    input_df['age'] = age_encoder.transform(input_df['age'])

    # Ensure 'generation' column exists
    if target not in data.columns:
        raise KeyError(f"Missing required target column in dataset: {target}")

    # Split dataset into features and target
    X = data[features]
    y = data[target]

    # Initialize and train the model on the whole dataset
    model = RandomForestClassifier()
    model.fit(X, y)

    # Make prediction for the input sample
    prediction_score = model.predict(input_df[features])

    # Return the predicted value
    print("Predicted value for input:", prediction_score[0])
    return prediction_score[0]  # Return the prediction result

# Usage example:
# message = {'year': 1987, 'country': 'Albania', 'age': '15-24 years', 'generation': 'Generation X'}
# prediction = predict_from_message(file_path, message)
# print(f"Predicted generation: {prediction}")
