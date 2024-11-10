# Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import streamlit as st

# Set the environment variable for the number of threads before importing sklearn
os.environ["OMP_NUM_THREADS"] = "5"  # Adjust this if necessary

# Streamlit App Title
st.title("Customer Analysis Dashboard")

# Load datasets
df1 = pd.read_csv("data/cleaned_customer_demographics.csv")
df2 = pd.read_csv("data/cleaned_customer_transactions.csv")
df3 = pd.read_csv("data/cleaned_social_media_interaction.csv")

# Display initial datasets for exploration
st.subheader("Customer Demographics Data")
st.write(df1.head())

st.subheader("Customer Transactions Data")
st.write(df2.head())

st.subheader("Social Media Interactions Data")
st.write(df3.head())

# Merge Customer Demographics with Customer Transactions
merged_df = pd.merge(df1, df2, on='Customer ID', how='inner')
merged_df = pd.merge(merged_df, df3, on='Customer ID', how='inner')

# Check for missing values
missing_values_final = merged_df.isnull().sum()
st.subheader("Missing Values in Final Merged Data")
st.write(missing_values_final[missing_values_final > 0])

# Convert date columns
for date_col in ['Sign Up Date', 'Transaction Date', 'Interaction Date']:
    if date_col in merged_df.columns:
        merged_df[date_col] = pd.to_datetime(merged_df[date_col], errors='coerce')

# Calculate RFM metrics
latest_date = merged_df['Transaction Date'].max()
reference_date = latest_date + pd.Timedelta(days=1)

rfm_df = merged_df.groupby('Customer ID').agg({
    'Transaction Date': lambda x: (reference_date - x.max()).days,
    'Customer ID': 'count',
    'Amount': 'sum'
}).rename(columns={
    'Transaction Date': 'Recency',
    'Customer ID': 'Frequency',
    'Amount': 'Monetary'
}).reset_index()

# Log transformation of RFM values
rfm_df['Recency_log'] = np.log1p(rfm_df['Recency'])
rfm_df['Frequency_log'] = np.log1p(rfm_df['Frequency'])
rfm_df['Monetary_log'] = np.log1p(rfm_df['Monetary'])

st.subheader("RFM Data")
st.write(rfm_df.head())

# Plotting RFM Distributions
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(rfm_df['Recency'], bins=20, kde=True, ax=ax[0])
ax[0].set_title("Recency Distribution")
sns.histplot(rfm_df['Frequency'], bins=20, kde=True, ax=ax[1])
ax[1].set_title("Frequency Distribution")
sns.histplot(rfm_df['Monetary'], bins=20, kde=True, ax=ax[2])
ax[2].set_title("Monetary Distribution")
st.pyplot(fig)

# Elbow Method for Optimal K (KMeans Clustering)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_df[['Recency_log', 'Frequency_log', 'Monetary_log']])
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_title('Elbow Method For Optimal K')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Cluster the data
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_df[['Recency_log', 'Frequency_log', 'Monetary_log']])

st.subheader("Clustered RFM Data")
st.write(rfm_df.head())

# Visualizing Clusters with Seaborn Pairplot
fig = sns.pairplot(rfm_df, hue='Cluster', palette='viridis', corner=True)
st.pyplot(fig)

# Train-Test Split for Model Evaluation
X = rfm_df[['Recency', 'Frequency', 'Monetary']]
y = rfm_df['Monetary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Pipeline
model = Pipeline(steps=[
    ('scaler', StandardScaler()), 
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Training and Prediction
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics Calculation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation Metrics")
st.write(f"Mean Absolute Error: {mae}")
st.write(f"R² Score: {r2}")

# Plotting Model Predictions vs Actual
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_title("Predicted vs Actual Monetary Values")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
st.pyplot(fig)

# Additional Features: Transaction Frequency, Avg. Transaction Value, Last Purchase Date
transaction_counts = df2.groupby('Customer ID')['Transaction Date'].count().reset_index(name='Transaction Frequency')
average_transaction_value = df2.groupby('Customer ID')['Amount'].mean().reset_index(name='Average Transaction Value')
last_purchase_date = df2.groupby('Customer ID')['Transaction Date'].max().reset_index(name='Last Purchase Date')

# Merge with merged_df
merged_df = merged_df.merge(transaction_counts, on='Customer ID', how='left')
merged_df = merged_df.merge(average_transaction_value, on='Customer ID', how='left')
merged_df = merged_df.merge(last_purchase_date, on='Customer ID', how='left')

# Time Since Last Purchase
merged_df['Last Purchase Date'] = pd.to_datetime(merged_df['Last Purchase Date'])
merged_df['Time Since Last Purchase'] = (pd.to_datetime('today') - merged_df['Last Purchase Date']).dt.days

st.subheader("Merged Data with Additional Features")
st.write(merged_df.head())

# Correlation Heatmap for Numeric Features Only
numeric_df = merged_df.select_dtypes(include=[np.number])  # Select only numeric columns
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title("Correlation Heatmap of Numeric Features")
st.pyplot(fig)
