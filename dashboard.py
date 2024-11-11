import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Setting up the page title and description
st.title("Customer Lifetime Value (CLV) Prediction & Segmentation Dashboard")
st.write("This dashboard provides customer segmentation and CLV prediction insights using Machine Learning.")

# Load data files
df_demographics = pd.read_csv("data/cleaned_customer_demographics.csv")
df_transactions = pd.read_csv("data/cleaned_customer_transactions.csv")
df_social_media = pd.read_csv("data/cleaned_social_media_interaction.csv")

# Merging Data
merged_df = pd.merge(df_demographics, df_transactions, on='Customer ID', how='inner')
merged_df = pd.merge(merged_df, df_social_media, on='Customer ID', how='inner')

# Display merged data preview
st.write("### Merged Data Preview")
st.dataframe(merged_df.head())

# Handling Missing Values
st.write("### Missing Values in Dataset")
missing_values = merged_df.isnull().sum()
st.write(missing_values[missing_values > 0])

# Date Parsing and Recency Calculation
merged_df['Transaction Date'] = pd.to_datetime(merged_df['Transaction Date'], errors='coerce')
reference_date = merged_df['Transaction Date'].max()
rfm_df = merged_df.groupby('Customer ID').agg({
    'Transaction Date': lambda x: (reference_date - x.max()).days,
    'Customer ID': 'count',
    'Amount': 'sum'
}).rename(columns={
    'Transaction Date': 'Recency',
    'Customer ID': 'Frequency',
    'Amount': 'Monetary'
})

# Log Transformations
rfm_df['Recency_log'] = np.log1p(rfm_df['Recency'])
rfm_df['Frequency_log'] = np.log1p(rfm_df['Frequency'])
rfm_df['Monetary_log'] = np.log1p(rfm_df['Monetary'])

st.write("### RFM Table with Log Transformations")
st.dataframe(rfm_df.head())

# Clustering and Visualization
st.write("### Clustering and Visualization")
rfm_features = rfm_df[['Recency_log', 'Frequency_log', 'Monetary_log']]

# Elbow Method for optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_features)
    wcss.append(kmeans.inertia_)

# Elbow Plot
fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
ax.set_title('Elbow Method for Optimal Clusters')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Clustering with Optimal Cluster Number
optimal_clusters = st.slider("Select Optimal Number of Clusters", 1, 10, 4)
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_features)

# Cluster Visualization
fig, ax = plt.subplots()
scatter = ax.scatter(rfm_df['Recency_log'], rfm_df['Monetary_log'], c=rfm_df['Cluster'], cmap='viridis')
ax.set_title('Customer Segmentation based on RFM')
ax.set_xlabel('Log-Recency')
ax.set_ylabel('Log-Monetary')
plt.colorbar(scatter, ax=ax)
st.pyplot(fig)

# Model Training for CLV Prediction
st.write("### CLV Prediction Model")
X = rfm_df[['Recency', 'Frequency', 'Monetary']]
y = rfm_df['Monetary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Model Performance:**")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Streamlit insights for each customer cluster
st.write("### Customer Segmentation Insights")
for cluster in sorted(rfm_df['Cluster'].unique()):
    st.write(f"**Cluster {cluster}:**")
    cluster_data = rfm_df[rfm_df['Cluster'] == cluster]
    st.write(f"Count: {len(cluster_data)}")
    st.write(f"Average Recency: {cluster_data['Recency'].mean():.2f}")
    st.write(f"Average Frequency: {cluster_data['Frequency'].mean():.2f}")
    st.write(f"Average Monetary: {cluster_data['Monetary'].mean():.2f}")
