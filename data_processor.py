import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os


#

def fetch_data(csv_file_path):
    """Fetch data from csv file"""
    try:
        # Check if file exists
        if not os.path.exists(csv_file_path):
            print(f"CSV file not found: {csv_file_path}")
            return None
    
        # Read CSV file
        data = pd.read_csv(csv_file_path)
        print(f"Successfully loaded {len(data)} transactions from {csv_file_path}.")
        return data
    
    except Exception as e:
        print(f"Error loading data from csv:{str(e)}")
        return None
    

    
def preprocess_data(data):
    """preprocess the transaction data with consistency checks.""" 
    if data is None:
        print("No data available, please fetch data first.")
        return None

    print("starting Data Preprocessing...")   
    df = data.copy()


    # Check for duplicates  
    print("Checking for duplicates...")
    total_duplicates = df.duplicated().sum()
    id_duplicates = df['TransactionID'].duplicated().sum()
    print(f"Duplicate transactions: {total_duplicates}")
    print(f"Duplicate TransactionID: {id_duplicates}")
    df = df.drop_duplicates(subset=['TransactionID'])
    

    # Data Consistency checks
    print("Checking data consistency...")
    unique_customers = df['CustomerID'].nunique()
    print(f"unique customers: {unique_customers}")

    #Check for invalid dates
    print("Checking date ranges...")
    df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], errors='coerce')

    # Handle missing values

    df = df.dropna()

    # Ensure proper datetime formate
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
    return df

def calculate_rfm_metrics(data):
    """Calculate RFM metrics"""
    if data is None:
        print("No data available, please fetch and preprocess data first.")
        return
    
    reference_date = data['TransactionDate'].max() + pd.Timedelta(days=1)

    rfm_df = data.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (reference_date - x.max()).days,
        'TransactionID': 'count',
         'TransactionAmount': 'sum'
    }).reset_index()

    rfm_df.columns= ['CustomerID', 'Recency','Frequency', 'Monetary']


    # Add Customer Demographics
    customer_dem = data.groupby('CustomerID').agg({
        'CustomerDOB': 'first',
        'CustGender': 'first',
        'CustLocation': 'first',
        'CustAccountBalance':'last'
    }).reset_index()

    rfm_df = rfm_df.merge( customer_dem, on='CustomerID', how='left')

    return rfm_df