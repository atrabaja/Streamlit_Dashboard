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
st.title("Customer Lifetime Value Prediction & Customer Segmentation Dashboard")
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

# Sliders to adjust Recency_log, Frequency_log, and Monetary_log
st.write("### Adjust RFM Log Values")
col1, col2, col3 = st.columns(3)
with col1:
    recency_log_adjust = st.slider("Recency Log", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
with col2:
    frequency_log_adjust = st.slider("Frequency Log", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
with col3:
    monetary_log_adjust = st.slider("Monetary Log", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Apply adjustments to RFM log values
rfm_df['Recency_log'] = np.log1p(rfm_df['Recency']) * recency_log_adjust
rfm_df['Frequency_log'] = np.log1p(rfm_df['Frequency']) * frequency_log_adjust
rfm_df['Monetary_log'] = np.log1p(rfm_df['Monetary']) * monetary_log_adjust

# Interactive customer selection
customer_id = st.selectbox("Select a Customer ID for detailed analysis:", rfm_df.index)
selected_customer_data = rfm_df.loc[customer_id]

st.write("### RFM Table with Log Transformations")
st.dataframe(rfm_df.head())

# Clustering with fixed 4 clusters
rfm_features = rfm_df[['Recency_log', 'Frequency_log', 'Monetary_log']]
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_features) + 1  # Assigning cluster numbers as 1, 2, 3, 4

# Filter to display the selected customer's cluster details
selected_customer_cluster = rfm_df.loc[customer_id, 'Cluster']
filtered_df = rfm_df[rfm_df['Cluster'] == selected_customer_cluster]

# Dropdown for X and Y axis selection in scatter plot
st.write("### Customer Segmentation based on RFM")
x_axis_option = st.selectbox("Select X-axis feature for Scatter Plot:", ['Recency_log', 'Frequency_log', 'Monetary_log'])
y_axis_option = st.selectbox("Select Y-axis feature for Scatter Plot:", ['Recency_log', 'Frequency_log', 'Monetary_log'])

# Scatter Plot for Clusters
fig, ax = plt.subplots()
scatter = ax.scatter(rfm_df[x_axis_option], rfm_df[y_axis_option], c=rfm_df['Cluster'], cmap='viridis', alpha=0.6)
ax.scatter(selected_customer_data[x_axis_option], selected_customer_data[y_axis_option], color='red', label='Selected Customer', s=100, edgecolor='black')
ax.set_title('')
ax.set_xlabel(x_axis_option)
ax.set_ylabel(y_axis_option)
plt.colorbar(scatter, ax=ax)
plt.legend()
st.pyplot(fig)

# Display updated Customer Segmentation Insights table based on selected customer
st.write(f"### Customer Segmentation Insights for Cluster {selected_customer_cluster}")
cluster_insights = filtered_df.groupby('Cluster').agg(
    Customer_Count=('Recency', 'size'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).reset_index()

# Display insights as an interactive table
st.dataframe(cluster_insights)

# CLV Prediction Model
st.write("### CLV Prediction Result")

# Train-test split and model training based on new segmentation
X = rfm_df[['Recency', 'Frequency', 'Monetary']]
y = rfm_df['Monetary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Re-train the model based on updated clusters
model.fit(X_train, y_train)
rfm_df['Predicted_CLV'] = model.predict(X)

# Adding segmentation labels based on CLV
rfm_df['Segment'] = pd.qcut(rfm_df['Predicted_CLV'], q=4, labels=['Low Value', 'Mid Value', 'High Value', 'Top Value'])

# Display CLV Score and Segmentation Label for the selected customer
predicted_clv = rfm_df.loc[customer_id, 'Predicted_CLV']
segment_label = rfm_df.loc[customer_id, 'Segment']
st.write(f"**Predicted CLV Score for Customer {customer_id}:** ${predicted_clv:.2f}")
st.write(f"**Customer Segmentation Label:** {segment_label}")

# Actionable Insights based on Segmentation Label
if segment_label == 'Top Value':
    st.write("**Actionable Insights:** Focus on retention strategies, loyalty rewards, and exclusive offers to maintain engagement.")
elif segment_label == 'High Value':
    st.write("**Actionable Insights:** Consider upselling or cross-selling opportunities with personalized recommendations.")
elif segment_label == 'Mid Value':
    st.write("**Actionable Insights:** Engage with targeted promotions to increase purchase frequency.")
else:
    st.write("**Actionable Insights:** Re-engagement campaigns and personalized offers may help retain this customer.")
    
# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Model Performance:**")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")
