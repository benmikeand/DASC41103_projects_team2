import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# process and standardize data
def preprocess_data(df):

    # Drop the first column (seems like just a random index) and first row (column names)
    df = df.iloc[1:, 1:]
    
    # correct column names
    new_column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
    df.columns = new_column_names

    # Handle missing values
    df.replace('?', 'Missing', inplace=True)  # Letting missing values be their own class as they're only in categorical columns

    # Binarize the target variable
    df['class'] = df['class'].apply(lambda x: 1 if x == '>50K' else 0)

    # keep track of rows indexes to connect X and y
    df = df.reset_index(names=['original_index'])

    # Encode categorical features
    df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'sex'], dtype = int)

    # Change data types
    df = df.apply(pd.to_numeric, errors='coerce')

    # Separate features and target
    X = df.drop(columns=['class'], axis=1)
    y = df[['original_index','class']]

    # Standardize numerical features
    numeric_cols = ['age','fnlwgt','education-num','capital-gain', 'capital-loss', 'hours-per-week']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Display preprocessed data
    merged_df = pd.concat([X, y['class']], axis=1)
    display(merged_df.head())

    return X, y

# process and standardize data
def preprocess_validation_data(df):

    # Drop the first column (seems like just a random index) and first row (column names)
    df = df.iloc[1:, 1:]
    
    # correct column names
    new_column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    df.columns = new_column_names

    # Handle missing values
    df.replace('?', 'Missing', inplace=True)  # Letting missing values be their own class as they're only in categorical columns

    # keep track of rows indexes to connect X and y
    df = df.reset_index(names=['original_index'])
 
    # Encode categorical features
    df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'sex'], dtype = int)

    # change data types
    df = df.apply(pd.to_numeric, errors='coerce')

    # Separate features and target
    X = df

    # Standardize numerical features
    numeric_cols = ['age','fnlwgt','education-num','capital-gain', 'capital-loss', 'hours-per-week']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    display(X.head())

    return X