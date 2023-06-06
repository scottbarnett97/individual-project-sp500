import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np
import env
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

###################################### Split ###########################################


def split_data(df):
    '''
    Be sure to code it as train, validate, test = split_data(df)
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123, 
                                       )
    #This confirms and Validates my split.
    
    print(f'train -> {train.shape}, {round(train.shape[0]*100 / df.shape[0],2)}%')
    print(f'validate -> {validate.shape},{round(validate.shape[0]*100 / df.shape[0],2)}%')
    print(f'test -> {test.shape}, {round(test.shape[0]*100 / df.shape[0],2)}%')
    
    return train, validate, test

####################################### Prep #############################################

def prep_zillow(df):
    '''
    This function cleans up the data, renames columns, converts floats to ints, drops nulls, defines fips as counties, and removes outliers.
    '''
    # Rename columns
    df = df.rename(columns={'bedroomcnt': 'bedrooms',
                            'bathroomcnt': 'bathrooms',
                            'calculatedfinishedsquarefeet': 'area',
                            'taxvaluedollarcnt': 'taxvalue',
                            'fips': 'county',
                            'lotsizesquarefeet': 'lot'})

    # Dropping nulls
    df = df.dropna()
    # Make floats into ints
    make_ints = ['bedrooms', 'bathrooms', 'area','lot', 'taxvalue']
    for col in make_ints:
        df[col] = df[col].astype(int)
    # Define fips as county names
    df['county'] = df['county'].map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    # Remove outliers
    z_threshold = 3
    numerical_columns = ['bedrooms', 'bathrooms', 'area', 'taxvalue','lot']
    # Create an outlier mask using the z-scores
    z_scores = np.abs(stats.zscore(df[numerical_columns]))
    outlier_mask = (z_scores < z_threshold).all(axis=1)
    df = df[outlier_mask]
    # Drop the 'transactiondate' column
    df = df.drop('transactiondate', axis=1)
    df_dummy=pd.get_dummies(columns=['county'], data=df)
    df = pd.concat([df.county, df_dummy],axis=1)
    return df