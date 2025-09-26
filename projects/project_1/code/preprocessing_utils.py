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
    df.replace('?', np.nan, inplace=True)

    df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
    df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])
    df['native-country'] = df['native-country'].fillna(df['native-country'].mode()[0])

    print(f"Total missing values in the DataFrame: {df.isnull().sum().sum()}") 

    # Binarize the target variable
    df['class'] = df['class'].apply(lambda x: 1 if x == '>50K' else 0)

    # keep track of rows indexes to connect X and y
    df = df.reset_index(names=['original_index'])

    #Columns to encode
    columns_to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'sex']
     
    # Encode categorical features
    #df = pd.get_dummies(df, columns = columns_to_encode, dtype = int)

    
    # Apply LabelEncoder to each column
    label_encoders = {}
    for col in columns_to_encode:    
        le = LabelEncoder()    
        df[col] = le.fit_transform(df[col])    
        label_encoders[col] = le  

    # change data types
    df = df.apply(pd.to_numeric, errors='coerce')


    # Separate features and target
    X = df.drop(columns=['class'], axis=1)
    y = df[['original_index','class']]

    # Standardize numerical features
    numeric_cols = ['age','fnlwgt','education-num','capital-gain', 'capital-loss', 'hours-per-week']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    display(df.head())
    
    # Check means and standard deviations
    print("Means after scaling:\n", X[numeric_cols].mean())
    print("\nStandard deviations after scaling:\n", X[numeric_cols].std())

    return X, y

# process and standardize data
def preprocess_test_data(df):

    # Drop the first column (seems like just a random index) and first row (column names)
    df = df.iloc[1:, 1:]
    
    
    # correct column names
    new_column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    df.columns = new_column_names

    # Handle missing values
    df.replace('?', np.nan, inplace=True)

    df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
    df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])
    df['native-country'] = df['native-country'].fillna(df['native-country'].mode()[0])

    print(f"Total missing values in the DataFrame: {df.isnull().sum().sum()}") 

    # keep track of rows indexes to connect X and y
    df = df.reset_index(names=['original_index'])

    # Columns to encode
    columns_to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'sex']
     
    # Encode categorical features
    #df = pd.get_dummies(df, columns = columns_to_encode, dtype = int)

    
    # Apply LabelEncoder to each column
    label_encoders = {}
    for col in columns_to_encode:    
        le = LabelEncoder()    
        df[col] = le.fit_transform(df[col])    
        label_encoders[col] = le 

    # change data types
    df = df.apply(pd.to_numeric, errors='coerce')


    # Separate features and target
    X = df

    # Standardize numerical features
    numeric_cols = ['age','fnlwgt','education-num','capital-gain', 'capital-loss', 'hours-per-week']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    display(df.head())
    
    # Check means and standard deviations
    print("Means after scaling:\n", X[numeric_cols].mean())
    print("\nStandard deviations after scaling:\n", X[numeric_cols].std())

    return X