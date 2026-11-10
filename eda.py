# ==========================================
# CUSTOMER CHURN PREDICTION
# Step 1: Data Loading and Initial Exploration
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("dataset/Churn_Modelling.csv")

# Display basic information
print("\n========== DATASET OVERVIEW ==========")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Check Duplicate Records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Check Target Variable Distribution
print("\nCustomer Churn Distribution:")
print(df["Exited"].value_counts())

print("\nChurn Percentage:")
print(df["Exited"].value_counts(normalize=True) * 100)

# ==========================================
# END OF DATA EXPLORATION
# ==========================================
