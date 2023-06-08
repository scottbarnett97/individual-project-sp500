# imports 
import warnings
warnings.filterwarnings("ignore")
# Tabular data friends:
import pandas as pd
import numpy as np
import math
# Data viz:
import matplotlib.pyplot as plt
import seaborn as sns

# Data acquisition
from pydataset import data
import scipy.stats as stats
import env
import os

# Sklearn stuff:
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split

## Regression Models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

## Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# Project related
import wrangle
############################### exploration data visuals ##############################

def plot_regression(train):
    '''
    this feature brings in the lmplot for the counties to tax value
    '''
    sns.color_palette("magma")
    sns.lmplot(x="area", y="taxvalue", hue='county', data=train, scatter=True,scatter_kws={'alpha': 0.2}, line_kws={'linewidth': 3})
    plt.xlabel("Area in Sq.ft")
    plt.ylabel("Tax Value in USD millions")
    plt.title("Regression Plot: Area vs. Tax Value")
    plt.show()

def bed_chart(data):
    '''
    creates a scatter plot for bedrooms vs tax value
    '''
    # Specify the x and y variables
    x = data['bedrooms']
    y = data['taxvalue']
    # Create the scatter plot
    plt.scatter(x, y, alpha=0.5)
    # Set the labels and title
    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Tax Value in millions USD")
    plt.title("Number of Bedrooms vs. Tax Value")
    # Display the chart
    plt.show()
    
def bath_chart(data):
    '''
    creates a scatter plot for bathrooms vs tax value
    '''
    # Specify the x and y variables
    x = data['bathrooms']
    y = data['taxvalue']
    # Create the scatter plot
    plt.scatter(x, y, alpha=0.5)
    # Set the labels and title
    plt.xlabel("Number of Bathrooms")
    plt.ylabel("Tax Value in millions USD")
    plt.title("Number of Bathrooms vs. Tax Value")
    # Display the chart
    plt.show()
    
def lot_chart(data):
    '''
    creates a scatter plot for lot vs tax value
    '''
    # Specify the x and y variables
    x = data['lot']
    y = data['taxvalue']
    # Create the scatter plot
    plt.scatter(x, y, alpha=0.5)
    # Calculate the best-fit line
    slope, intercept = np.polyfit(x, y, 1)
    best_fit_line = slope * x + intercept
    # Plot the best-fit line
    plt.plot(x, best_fit_line, color='red')
    # Set the labels and title
    plt.xlabel("Lot in Sqft")
    plt.ylabel("Tax Value in millions USD")
    plt.title("Lot vs. Tax Value with Best-Fit Line")
    # Display the chart
    plt.show()

def get_distplot(train):
    # Plot the distribution of the target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(train['taxvalue'], kde=True)
    plt.xlabel('Tax Value in millions USD')
    plt.ylabel('Count')
    plt.title('Distribution of Tax Values')
    # Add a vertical line for the baseline RMSE
    plt.axvline(x=383891.952694, color='red', linestyle='--', label='Baseline RMSE')
    plt.legend()
    plt.show()   
###############################  satatistical tests ######################

def run_low_ttest(data):
    '''
    runs a Ttest for low vs target
    '''
    x = data['low']
    y = data['target']
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(x, y)
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"
# Create a DataFrame to store the results
    results = pd.DataFrame({
        'T-Statistic': [t_statistic],
        'P-Value': [p_value],
        'Decision': [decision]})
    return results

def run_high_ttest(data):
    '''
    runs a Ttest for high vs target
    provides T-stat and P-value
    recomends rejection of null or not
    '''
    x = data['high']
    y = data['target']
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(x, y)
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"
# Create a DataFrame to store the results
    results = pd.DataFrame({
        'T-Statistic': [t_statistic],
        'P-Value': [p_value],
        'Decision': [decision]})
    return results

def run_open_ttest(data):
    '''
    runs a Ttest for open vs target
    provides T-stat and P-value
    recomends rejection of null or not
    '''
    x = data['open']
    y = data['target']
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(x, y)
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"
# Create a DataFrame to store the results
    results = pd.DataFrame({
        'T-Statistic': [t_statistic],
        'P-Value': [p_value],
        'Decision': [decision]})
    return results

def run_volume_ttest(data):
    '''
    runs a Ttest for volume vs target
    provides T-stat and P-value
    recomends rejection of null or not
    '''
    x = data['volume']
    y = data['target']
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(x, y)
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"
# Create a DataFrame to store the results
    results = pd.DataFrame({
        'T-Statistic': [t_statistic],
        'P-Value': [p_value],
        'Decision': [decision]})
    return results

def run_close_ttest(data):
    '''
    runs a Ttest for close vs target
    provides T-stat and P-value
    recomends rejection of null or not
    '''
    x = data['close']
    y = data['target']
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(x, y)
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"
# Create a DataFrame to store the results
    results = pd.DataFrame({
        'T-Statistic': [t_statistic],
        'P-Value': [p_value],
        'Decision': [decision]})
    return results

def run_tomorrow_ttest(data):
    '''
    runs a Ttest for tomorrow vs target
    provides T-stat and P-value
    recomends rejection of null or not
    '''
    x = data['tomorrow']
    y = data['target']
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(x, y)
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"
# Create a DataFrame to store the results
    results = pd.DataFrame({
        'T-Statistic': [t_statistic],
        'P-Value': [p_value],
        'Decision': [decision]})
    return results


        ################################ Modeling ####################################


def scale_data(train, validate, test, columns):
    """
    Scale the selected columns in the train, validate, and test data.
    Args:
        train (pd.DataFrame): Training data.
        validate (pd.DataFrame): Validation data.
        test (pd.DataFrame): Test data.
        columns (list): List of column names to scale.
    Returns:
        tuple: Scaled data as (X_train_scaled, X_validate_scaled, X_test_scaled).
    Tu run paste: X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test = scale_data(train, validate, test, ['open', 'high','low','close','volume'])

    """
    # create X & y version of train, where y is a series with just the target variable and X are all the features.
    X_train = train.drop(['target','date'], axis=1)
    y_train = train['target']
    X_validate = validate.drop(['target','date'], axis=1)
    y_validate = validate['target']
    X_test = test.drop(['target','date'], axis=1)
    y_test = test['target']
    # Create a scaler object
    scaler = MinMaxScaler()
    # Fit the scaler on the training data for the selected columns
    scaler.fit(X_train[columns])
    # Apply scaling to the selected columns in all data splits
    X_train_scaled = X_train.copy()
    X_train_scaled[columns] = scaler.transform(X_train[columns])

    X_validate_scaled = X_validate.copy()
    X_validate_scaled[columns] = scaler.transform(X_validate[columns])

    X_test_scaled = X_test.copy()
    X_test_scaled[columns] = scaler.transform(X_test[columns])
    return X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test


def get_baseline(y_train):
    '''
    this function returns a baseline for precision
    To run paste: explore.get_baseline(y_train)
    '''
    baseline_prediction = y_train.median()
    # Predict the majority class in the training set
    baseline_pred = [baseline_prediction] * len(y_train)
    precision = precision_score(y_train, baseline_pred)
    baseline_results = {'Baseline': [baseline_prediction],'Metric': ['Precision'], 'Score': [precision]}
    baseline_df = pd.DataFrame(data=baseline_results)
    return baseline_df



def create_models(seed=123):
    '''
    Create a list of machine learning models.
            Parameters: seed (integer): random seed of the models
            Returns: models (list): list containing the models
    This includes best fit hyperparamaenters    
    To run paste: explore.create_models(seed=123)
    '''
    models = []
    models.append(('k_nearest_neighbors', KNeighborsClassifier(n_neighbors=100)))
    models.append(('logistic_regression', LogisticRegression(random_state=seed)))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=3, min_samples_split=4, random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(max_depth=3,random_state=seed)))
    return models

 
def get_models(train, validate, test):
    '''
    This function runs all the models created by the create models function on train and validate df's
    and returns the precision for each in a pddataframe
    To run paste: results,X_train_scaled, X_test_scaled,y_test,y_train= explore.get_models(train, validate, test)
results
    '''
    # create models list
    models = create_models(seed=123)
    X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test = scale_data(train, validate, test, ['open', 'high','low','close','volume'])
    # initialize results dataframe
    results = pd.DataFrame(columns=['model', 'set', 'precision'])
    
    # loop through models and fit/predict on train and validate sets
    for name, model in models:
        # fit the model with the training data
        model.fit(X_train_scaled, y_train)
        
        # make predictions with the training data
        train_predictions = model.predict(X_train_scaled)
        
        # calculate training precision 
        train_precision = precision_score(y_train, train_predictions)
           
        # make predictions with the validation data
        val_predictions = model.predict(X_validate_scaled)
        
        # calculate validation precision
        val_precision = precision_score(y_validate, val_predictions)
        
        # append results to dataframe
        results = results.append({'model': name, 'set': 'train', 'precision': train_precision}, ignore_index=True)
        results = results.append({'model': name, 'set': 'validate', 'precision': val_precision}, ignore_index=True)

    return results,X_train_scaled, X_test_scaled,y_test,y_train