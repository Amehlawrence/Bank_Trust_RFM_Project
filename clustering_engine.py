import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Global Variables
kmeans = None
rfm_data = None


def find_optimal_clusters(rfm_scaled_df):
    """Determine optimal number of clusters"""

    WCSS = []
    silhouette_scores = []
    cluster_range = list(range(2, 11))  # try k = 2..10

    for k in cluster_range:
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_model.fit_predict(rfm_scaled_df)

        # Elbow: inertia
        WCSS.append(kmeans_model.inertia_)

        # Silhouette score (valid for k >= 2; our range already starts at 2)
        #if k > 1:
        score = silhouette_score(
            rfm_scaled_df, 
            kmeans_model.labels_  
        )
        silhouette_scores.append(score)

    best_k = cluster_range[np.argmax(silhouette_scores)]

    return {
        "cluster_range": cluster_range,
        "wcss": WCSS,
        "silhouette_scores" : silhouette_scores,
        "optimal_k": best_k,
        "best_silhouette" : max(silhouette_scores)

    }   


def apply_clustering(rfm_scaled_df, rfm_data, optimal_k = None):
    """Apply K-means clustering cleanly without global"""

    if optimal_k is None:
        results = find_optimal_clusters(rfm_scaled_df)
        optimal_k = results["optimal_k"]

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm_data = rfm_data.copy()
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled_df)

    #ANALYZE CLUSTERS

    cluster_analysis = rfm_data.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'R_score': 'mean',
        'F_score': 'mean',
        'M_score': 'mean',
        'CustomerID': 'count',
        'CustAccountBalance': 'mean',
        'CustGender': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    }).round(2)

    cluster_analysis.columns = [
        'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary',
        'Avg_R_score', 'Avg_F_score', 'Avg_M_score',
        'Customer_Count', 'Avg_Account_Balance', 'Most_Common_Gender'
    ]

    return cluster_analysis, rfm_data, optimal_k



def assign_cluster_name(stats):
    recency = stats['Avg_Recency']
    frequency = stats['Avg_Frequency']
    monetary = stats['Avg_Monetary']

    # 1) Long-dormant customers
    if recency > 365:
        # Haven't purchased in a year
        if monetary > 20000:
            return "High-value Dormant Customers"
        else:
            return "Dormant Low-value Customers"

    # 2) Inactive (6–12 months)
    elif recency > 180:
        # Haven't purchased in 6–12 months
        if monetary > 20000:
            return "High-value Inactive Customers"
        else:
            return "Inactive Low-value Customers"

    # 3) More recent customers (recency <= 180)
    else:
        if frequency > 10 and monetary > 30000:
            return "Active VIP Customers"
        elif frequency > 8 and monetary < 10000:
            return "Active Low-value Customers"
        else:
            return "Regular Customers"
        
def assign_cluster_names(rfm_data, cluster_analysis):
    rfm_data = rfm_data.copy()
    rfm_data['Cluster_Name'] = rfm_data['Cluster'].map(
         lambda x: assign_cluster_name(cluster_analysis.loc[x])
    )
    return rfm_data