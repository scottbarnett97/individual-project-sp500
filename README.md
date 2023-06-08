# Individual-Project- S&P 500 up or down

## Description
* We want to be able to predict the wheter the S&P 500 will be up or down by close on a geven day

## Background Info
* The S&P 500 is a stock market index composed of 500 large companies having common stock listed on the New York Stock Exchange (NYSE) or Nasdaq. Founded in 1923, it is now considered one of the best overall indicators of the US stock market.
* There are several investment vehicles offered by various companies that aim to mimic the composition of the S&P 500 and are tradeble under ticker symbols like SPY and SPX
* For this project we will use ^GSPC, which is Yahoo Finance's 'proprietary' ticker for the S&P 500 index. 
    * However, it must be remembered that ^GSPC is a price index and is not tradeable. It only shows the movement of stock prices in the S&P 500 index.

## Goals¶
* Construct an ML model that predicts S&P 500 Up or Down
* Find the key drivers of S&P 500 movement.
* Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
* Make recommendations on what works or doesn't work in predicting the S&P 500 movement up or down.

## Initial hypotheses and/or questions you have of the data, ideas
There should be some combination of features that can be used to build a predictive model wheter the S&P 500 will be up or down on a given day
* 1. ?
* 2. ? 
* 3. ? 
* 4. ? 
*****************************************
## Project Plan 
* Acquire data from Yahoo Finance 
* Prepare data
    * Drop data prior to 1990 
        * Modern electronic trading dramatically chaged access to the markets
        * This caused a major change in market behaviors 
        * dropping this data will keep the model from favoring 70 years of data that generally no longer apply 
        * NASDAQ introduced electronic trading inthe 1980's, but the NYSE didn't begin using them until the 90's
    * drop last row containing null value as that is what we are trying to predict 
    * Create Engineered columns from existing data
        * created tommorrw column based on the close price for today 
        * created a target colum that is the encoded up(1) or down(0) value 1 indicating a day to buy stock with a predicted rise in price the next day

*** Explore data in search of drivers of S&P 500 movement
    **** Answer the following initial questions
        ***** 1. ?
        ***** 2. ?
        ***** 3. ? 
        ***** 4. ? 
*** Develop a model to predict what day to buy stock on S&P500 movement up or down
*** Draw conclusions

## Data Dictionary

| Feature | Datatype | Key | Definition |
|---|---|---|---|
| Date | object | Unique | Date of trading activity |
| High | float64 | in USD | The highest value reached that day |
| Low | float64 | in USD | The lowest value reached that day |
| Close | float64 | in USD | The value at the time the market opened |
| Open | float64 | in USD | The value at the time the market opened |
| Volume | int64 | count | The total number of shares traded that day |
| Tomorrow | float64 | in USD | The closing price the next day |
| Target | int64 | 1 = Yes  <br>0 = No | Day you should buy because the value is expected to rise the next day |
## Steps to Reproduce
* 1. Clone this repo
* 2. Acquire the data from yfinance 
    * use terminal to run pip install yfinance 
    * add import yfinance as yf to your libraries
* 3. Put the data in the file containing the cloned repo
* 4. Run notebook
* 5. Note: Your impored csv. will contain data between my import date and yours
    * Option 1: you can delete the additional rows to match 
    * Option 2: accept that they are there and may cause minute changes to results- these should be insignificant overall

## Takeaways and Conclusions
* Of the features examined all proved relevant to predicting tax values
* * RMSE proved the best metric for evaluating various models created 
* By combining the features into a series of models I found the Polynomial Regression model was best
* On the test set the model returned a RMSE of 307698.389898 
    * this out performs the baseline RMSE 383891.952694   
* This model could be used in production, but I belive it could be improved further

# Recomendations
* Continue developing the model
* Property values are determined by a large variety of factors so more features may improve the model
# Next Steps
* If provided more time to work on the project I would want to explore more features to develop a better model