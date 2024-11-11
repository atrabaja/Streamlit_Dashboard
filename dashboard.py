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

# Set up the page title and description
st.title("Interactive CLV Prediction & Customer Segmentation Dashboard")
st.write("Explore customer segments and predict Customer Lifetime Value (CLV) with an interactive dashboard.")

# Load data
df1 = pd.read_csv("data/cleaned_customer_demographics.csv")
df2 = pd.read_csv("data/cleaned_customer_transactions.csv")
df3 = pd.read_csv("data/cleaned_social_media_interaction.csv")

# Data Merging
merged_df = pd.merge(df1, df2, on='Customer ID', how='inner')
merged_df = pd.merge(merged_df, df3, on='Customer ID', how='inner')

# Display merged data preview
st.write("### Merged Data Preview")
st.dataframe(merged_df.head())

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

# Interactive Section for Clustering and Visualizations
st.write("### Clustering and Visualization")

# Dropdown for cluster features
selected_feature_x = st.selectbox("Select X-axis feature for Scatter Plot:", ['Recency_log', 'Frequency_log', 'Monetary_log'])
selected_feature_y = st.selectbox("Select Y-axis feature for Scatter Plot:", ['Recency_log', 'Frequency_log', 'Monetary_log'])

# Histogram: Distribution of selected feature
st.write("### Histogram")
selected_feature_hist = st.selectbox("Select Feature for Histogram:", ['Recency', 'Frequency', 'Monetary'])
fig, ax = plt.subplots()
sns.histplot(rfm_df[selected_feature_hist], bins=20, kde=True, ax=ax)
ax.set_title(f"Distribution of {selected_feature_hist}")
st.pyplot(fig)

# Line chart to show cluster count vs. WCSS (Elbow Method)
st.write("### Line Chart: Elbow Method")
wcss = []
rfm_features = rfm_df[['Recency_log', 'Frequency_log', 'Monetary_log']]
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_features)
    wcss.append(kmeans.inertia_)

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

# Scatter Plot for Clusters
st.write("### Scatter Plot: Customer Segmentation")
fig, ax = plt.subplots()
scatter = ax.scatter(rfm_df[selected_feature_x], rfm_df[selected_feature_y], c=rfm_df['Cluster'], cmap='viridis')
ax.set_title('Customer Segmentation based on RFM')
ax.set_xlabel(selected_feature_x)
ax.set_ylabel(selected_feature_y)
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
