import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def calculate_rfm_scores(rfm_data):

    if rfm_data is None:
        print("No RFM data available for scoring.")
        return None
    
    print("calculating RFM Scores and segmentation...")

    #Recency: lower recency = better(more recent)
    rfm_data['R_score'] = pd.qcut(rfm_data['Recency'], q=5, labels=[5, 4, 3, 2, 1])

#Frequency: higher frequency = better
    rfm_data['F_score'] = pd.qcut(rfm_data['Frequency'], q=5, labels=[1, 2, 3, 4, 5])

#Monetary:Higher monetary = better
    rfm_data['M_score'] = pd.qcut(rfm_data['Monetary'], q=5, labels=[1, 2, 3, 4, 5])

#Convert scores to numeric
    rfm_data[['R_score', 'F_score', 'M_score']] = rfm_data[['R_score', 'F_score','M_score']].astype(int)

    rfm_data['RFM_Score'] = rfm_data['R_score'] + rfm_data['F_score']  + rfm_data['M_score']

    rfm_data['RFM_Group'] = (
        rfm_data['R_score'].astype(str) 
        + rfm_data['F_score'].astype(str) 
        + rfm_data['M_score'].astype(str)
    )

    print("RFM Scoring and segmentation completed successfully.")
    print(f"Average RFM Score: {rfm_data['RFM_Score'].mean():.2f}")


    return rfm_data

    
def prepare_for_clustering(rfm_data):
    """Prepare RFM data for clustering"""
    if rfm_data is None:
        print("No RFM data available for clustering preparation.")
        return None

    print("Preparing RFM data for clustering...")


    # Extract key columns
    rfm_clustering = rfm_data[['Recency', 'Frequency', 'Monetary']].copy()


    # Log transformation to reduce skewness   
    # Handle skewness with log transformation
    rfm_clustering['Recency'] = np.log1p(rfm_clustering['Recency'])
    rfm_clustering['Frequency'] = np.log1p(rfm_clustering['Frequency'])
    rfm_clustering['Monetary'] = np.log1p(rfm_clustering['Monetary'])

#Standardize the Feature (Scaling)
    rfm_scaled = scaler.fit_transform(rfm_clustering)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

    print("RFM data successfully scaled and transformed for clustering.")
    return rfm_scaled_df 