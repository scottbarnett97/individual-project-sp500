import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
import os
 

####################### Imports ############################

def check_file_exists(fn, query, url):
    '''
    this function checks to see if the .csv file already exists. If yes it reads it
    '''
    if os.path.isfile(fn):
        print('csv file found and loaded\n')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv\n')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df 
    
def get_zillow_data():
    '''
    This function brings in the Zillow DF using mySQL from the Codeup server
    It uses the env.py file for access
    '''
    url = env.get_db_url('zillow')
    filename = 'zillow.csv'
    query = '''
        select taxvaluedollarcnt, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, fips,lotsizesquarefeet, transactiondate
        FROM properties_2017
        JOIN propertylandusetype USING (propertylandusetypeid)
        JOIN predictions_2017 USING (parcelid)
        WHERE propertylandusetypeid IN (261 , 279)
            ;
    '''        
    df = check_file_exists(filename, query, url)
    return df 

    
    
    
    
