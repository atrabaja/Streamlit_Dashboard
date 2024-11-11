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

# Custom CSS to set the background, font colors, and secondary color
st.markdown(
    """
    <style>
    /* Set background to white */
    .reportview-container {
        background: white;
    }
    .sidebar .sidebar-content {
        background: white;
    }

    /* Set font color to black */
    body, .markdown-text-container, .css-1d391kg p, .css-1h0j3t5, .css-qrbaxs, .css-1r6slb0, .css-2trqyj {
        color: black;
    }

    /* Primary color styling */
    .stButton>button {
        background-color: #289097;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stSelectbox, .stSlider {
        color: #289097;
    }

    /* Header adjustments */
    h1, h2, h3, h4, h5, h6 {
        color: #289097;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title and description
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

# Clustering with Fixed 4 Clusters
rfm_features = rfm_df[['Recency_log', 'Frequency_log', 'Monetary_log']]
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_features)

# Interactive Histogram
st.write("### Interactive Histogram")
selected_feature_hist = st.selectbox("Select Feature for Histogram:", ['Recency', 'Frequency', 'Monetary'])
fig, ax = plt.subplots()
sns.histplot(rfm_df[selected_feature_hist], bins=20, kde=True, ax=ax)
ax.set_title(f"Distribution of {selected_feature_hist}")
st.pyplot(fig)

# Scatter Plot for Clusters
st.write("### Interactive Scatter Plot: Customer Segmentation")
selected_feature_x = st.selectbox("Select X-axis feature for Scatter Plot:", ['Recency_log', 'Frequency_log', 'Monetary_log'])
selected_feature_y = st.selectbox("Select Y-axis feature for Scatter Plot:", ['Recency_log', 'Frequency_log', 'Monetary_log'])

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

# Cluster Insights Table
st.write("### Customer Segmentation Insights Table")
cluster_insights = rfm_df.groupby('Cluster').agg(
    Customer_Count=('Recency', 'size'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).reset_index()

# Display insights as an interactive table
st.dataframe(cluster_insights)
