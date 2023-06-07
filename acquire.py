import pandas as pd
import numpy as np
import env
import os
 

####################### Imports ############################

def get_sp500():
    if os.path.exists("sp500.csv"):
        sp500 = pd.read_csv("sp500.csv", index_col=0)
    else:
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        sp500.to_csv("sp500.csv")
    return sp500

    
    
    
    
