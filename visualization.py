import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Initialize visual style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def generate_cluster_profiles(rfm_data):
    """Generate summarized cluster profiles from RFM data"""

    cluster_profiles = rfm_data.groupby('Cluster_Name').agg({
        'Recency': ['mean', 'std'],
        'Frequency': ['mean', 'std'],
        'Monetary': ['mean', 'std'],
        'CustomerID': 'count',
        'CustAccountBalance': 'mean'
    }).round(2)

    # Flatten column names
    cluster_profiles.columns = ['_'.join(col).strip() for col in cluster_profiles.columns.values]

    cluster_profiles = cluster_profiles.rename(columns={
        'Recency_mean': 'Avg_Recency',
        'Recency_std': 'Std_Recency',
        'Frequency_mean': 'Avg_Frequency',
        'Frequency_std': 'Std_Frequency',
        'Monetary_mean': 'Avg_Monetary',
        'Monetary_std': 'Std_Monetary',
        'CustomerID_count': 'Customer_Count',
        'CustAccountBalance_mean': 'Avg_Account_Balance'
    })

    # Add percentage column
    cluster_profiles['Percentage'] = (
        cluster_profiles['Customer_Count'] / len(rfm_data) * 100
    ).round(2)

    return cluster_profiles


def plot_segmentation_distribution(rfm_data):
    """Generate pie chart of customer segmentation distribution"""

    segments_counts = rfm_data['Cluster_Name'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
              '#FFEAA7', '#DDA0DD', '#98D8C8']
    explode = [0.05] * len(segments_counts)

    fig, ax = plt.subplots(figsize=(14, 8))

    wedges, texts, autotexts = ax.pie(
        segments_counts.values,
        labels=segments_counts.index,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(segments_counts.values) / 100):,})',
        startangle=90,
        colors=colors[:len(segments_counts)],
        explode=explode,
        shadow=True
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('RFM Customer Distribution', fontsize=16, fontweight='bold')
    ax.axis('equal')
    plt.tight_layout()

    return fig


def plot_rfm_comparison(cluster_profiles):
    """Generate bar chart comparing RFM metrics across segments"""

    rfm_metrics = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(cluster_profiles))
    width = 0.25

    for i, metric in enumerate(rfm_metrics):
        values = cluster_profiles[metric].values
        if metric == 'Avg_Monetary':
            values = values / 1000  # convert to thousands

        ax.bar(x_pos + i * width, values, width=width,
               label=metric.replace('Avg_', ''), alpha=0.8)

    ax.set_xlabel('Customer Segments')
    ax.set_ylabel('Values (Monetary in thousands)')
    ax.set_title('Average RFM Metrics by Customer Segments')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(cluster_profiles.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_segment_sizes(cluster_profiles):
    """Generate horizontal bar chart comparing segment sizes"""

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
              '#FFEAA7', '#DDA0DD', '#98D8C8']

    segment_sizes = cluster_profiles.sort_values('Customer_Count')['Customer_Count']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        range(len(segment_sizes)),
        segment_sizes.values,
        color=colors[:len(segment_sizes)]
    )

    ax.set_yticks(range(len(segment_sizes)))
    ax.set_yticklabels(segment_sizes.index)
    ax.set_xlabel('Number of Customers')
    ax.set_title('Segment Size Comparison')

    for i, v in enumerate(segment_sizes.values):
        ax.text(v + 50, i, f'{v:,}', va='center', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_segment_revenue_percentage(rfm_data):
    """Generate a donut chart showing revenue contribution (%) by each segment"""

    revenue_per_segment = rfm_data.groupby('Cluster_Name')['Monetary'].sum()
    total_revenue = revenue_per_segment.sum()
    revenue_pct = (revenue_per_segment / total_revenue * 100).round(2)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
              '#FFEAA7', '#DDA0DD', '#98D8C8']

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts = ax.pie(
        revenue_pct.values,
        labels=[f"{v}%" for v in revenue_pct.values],
        colors=colors[:len(revenue_pct)],
        wedgeprops=dict(width=0.4),
    )

    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')

    ax.legend(
        wedges,
        revenue_pct.index,
        title="Customer Segments",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    ax.set_title("Revenue Contribution by Customer Segment",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig
