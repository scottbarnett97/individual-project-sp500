from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
 

####################### Acquire ############################

def get_sp500():
    '''
    this function checks to see if .csv exists
    if not it pulls in data creates .csv
    '''
    if os.path.exists("sp500.csv"):
        df = pd.read_csv("sp500.csv", index_col=0)
    else:
        df = yf.Ticker("^GSPC")
        df = df.history(period="max")
        df.to_csv("sp500.csv")
    return df

######################    Prepare  #############################
def prep_data(df):
    ''' this function preps the data set by:
    Creating a Tomorrow column as next day open projection based on close
    dropping unneeded columns 'Dividends' and 'Stock Split'
    setting a target dummy column so 0=down and 1=up 
    setting datetime index format
    '''
    # Selecting data from 1990 onwards
    df = df.loc["1990-01-01":].copy()
    #reset index
    df = df.reset_index()

    # Make column names lowercase
    df.columns = [col.lower() for col in df]
    #drop time
    df.date = pd.to_datetime(df.date,utc=True).dt.date
    # Creating Tomorrow as next day's open projection based on close
    df["tomorrow"] = df["close"].shift(-1)
    # Dropping unused columns
    df = df.drop(['stock splits', 'dividends'], axis=1) 
    # Setting up target column as an int, 0 for down days and 1 for up days
    df["target"] = (df["tomorrow"] > df["close"]).astype(int)
    df = df.drop(df.index[-1])
    return df    

    
################   Split   #####################################


def split_data(df):
    '''
    Be sure to code it as train, validate, test = split_data(df)
    take in a DataFrame and return train, validate, and test DataFrames; .
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

