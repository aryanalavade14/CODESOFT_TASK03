import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
df = pd.read_csv("dataset/Churn_Modelling.csv")

print("Original Dataset Shape:", df.shape)
df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1,
    inplace=True
)

print("After Removing Unnecessary Columns:", df.shape)

X = df.drop("Exited", axis=1)

y = df["Exited"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X.select_dtypes(include=["object", "str"]).columns

print("Numerical Features:", list(numerical_features))
print("Categorical Features:", list(categorical_features))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("Processed Training Shape:", X_train_processed.shape)
print("Processed Testing Shape:", X_test_processed.shape)

print("\nData Preprocessing Completed Successfully!")