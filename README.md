# Build An End To End Machine Learning Pipeline Using SageMaker 
This project focuses on implementing an end to end solution for a machine learning problem that predicts customer churn based on a streaming music app.

## Background

This particular challenge was originally introduced as a Kaggle competition in 2018. The goal was to build an algorithm that predicts whether a subscription user will churn using a donated dataset from KKBOX.

For a subscription business, accurately predicting churn is critical to long-term success.

Even slight variations in churn can drastically affect profits.

KKBOX is Asia’s leading music streaming service, holding the world’s most comprehensive Asia-Pop music library with over 30 million tracks. They offer a generous, unlimited version of their service to millions of people, supported by advertising and paid subscriptions. This delicate model is dependent on accurately predicting churn of their paid users.

In this project, we'll build a machine learning pipeline using XGBoost algorithm to predict whether a user will churn after their subscription expires. Currently, the company uses survival analysis techniques to determine the residual membership life time for each subscriber.

Once you understand the code structure described below, you can inspect the code and you can start customizing it for your own business case. This is only sample code, and you own this repository for your business use case. Please go ahead, modify the files, commit them and see the changes kick off the SageMaker pipelines in the CICD system.

## Data
We'll use the dataset provided by KKBox, a music streaming service made publically available by Kaggle [here](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data). The goal for the challenge is to predict whether a user will churn after his/her subscription expires. Specifically, we want to forecast if a user make a new service subscription transaction within 30 days after the current membership expiration date.

A brief description of the dataset used in this project:

### Churn Dataset
Containing the user ids and whether they have churned.

* msno: user id
* is_churn: This is the target variable. Churn is defined as whether the user did not continue the subscription within 30 days of expiration. is_churn = 1 means churn,is_churn = 0 means renewal.


### Transactions DataSet
Transactions of users through 2017.

* msno: user id
* payment_method_id: payment method
* payment_plan_days: length of membership plan in days
* plan_list_price: in New Taiwan Dollar (NTD)
* actual_amount_paid: in New Taiwan Dollar (NTD)
* is_auto_renew
* transaction_date: format %Y%m%d
* membership_expire_date: format %Y%m%d
* is_cancel: whether or not the user canceled the membership in this transaction.

### User Logs
Daily user logs describing listening behaviors of a user.

* msno: user id
* date: format %Y%m%d
* num_25: # of songs played less than 25% of the song length
* num_50: # of songs played between 25% to 50% of the song length
* num_75: # of songs played between 50% to 75% of of the song length
* num_985: # of songs played between 75% to 98.5% of the song length
* num_100: # of songs played over 98.5% of the song length
* num_unq: # of unique songs played
* total_secs: total seconds played
