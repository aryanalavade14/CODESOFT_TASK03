
# ============================================================
# CUSTOMER CHURN PREDICTION USING MACHINE LEARNING
# MODEL TRAINING AND EVALUATION
# ============================================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# STEP 2: LOAD DATASET

print("\n========== LOADING DATASET ==========")

df = pd.read_csv("dataset/Churn_Modelling.csv")

print("Dataset Loaded Successfully!")
print("Original Dataset Shape:", df.shape)


# STEP 3: DATA CLEANING

print("\n========== DATA CLEANING ==========")

# Remove unnecessary columns

df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1,
    inplace=True
)

print("After Removing Unnecessary Columns:", df.shape)


# STEP 4: SEPARATE FEATURES AND TARGET

X = df.drop("Exited", axis=1)

y = df["Exited"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# STEP 5: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns

print("\nNumerical Features:", list(numerical_features))

print("Categorical Features:", list(categorical_features))


# STEP 6: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLITTING ==========")

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# STEP 7: CREATE PREPROCESSING PIPELINE

preprocessor = ColumnTransformer(

    transformers=[

        ("num", StandardScaler(), numerical_features),

        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )

    ]
)

print("\nPreprocessing Pipeline Created Successfully!")


# STEP 8: DEFINE MACHINE LEARNING MODELS

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )

}


# STEP 9: TRAIN AND EVALUATE MODELS

results = []

best_model = None

best_score = -1

confusion_matrices = {}

print("\n========== MODEL TRAINING STARTED ==========")


for name, model in models.items():

    print("\nTraining:", name)

    # Create complete ML pipeline

    pipeline = Pipeline([

        ("preprocessor", preprocessor),

        ("classifier", model)

    ])

    # Train model

    pipeline.fit(X_train, y_train)

    # Predict customer churn

    y_pred = pipeline.predict(X_test)

    # Predict probabilities

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(y_test, y_prob)

    # Store results

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc

    })

    # Store confusion matrix

    cm = confusion_matrix(y_test, y_pred)

    confusion_matrices[name] = cm

    # Display model performance

    print("\n-----------------------------------")

    print("MODEL:", name)

    print("-----------------------------------")

    print("Accuracy :", round(accuracy, 4))

    print("Precision:", round(precision, 4))

    print("Recall   :", round(recall, 4))

    print("F1 Score :", round(f1, 4))

    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")

    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")

    print(cm)

    # Select best model using F1 score

    if f1 > best_score:

        best_score = f1

        best_model = pipeline

        best_model_name = name


# STEP 10: MODEL COMPARISON

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)


# STEP 11: SAVE MODEL COMPARISON

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")


# STEP 12: VISUALIZE MODEL PERFORMANCE

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1 Score"
)

plt.title("Machine Learning Model Comparison")

plt.xlabel("ML Algorithms")

plt.ylabel("F1 Score")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("model_comparison.png")

plt.show()


# STEP 13: CONFUSION MATRIX OF BEST MODEL

best_cm = confusion_matrices[best_model_name]

plt.figure(figsize=(6, 5))

sns.heatmap(
    best_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Exited"],
    yticklabels=["Stayed", "Exited"]
)

plt.title(f"Confusion Matrix - {best_model_name}")

plt.xlabel("Predicted Values")

plt.ylabel("Actual Values")

plt.tight_layout()

plt.savefig("best_model_confusion_matrix.png")

plt.show()


# STEP 14: SAVE BEST MODEL

joblib.dump(
    best_model,
    "best_churn_model.pkl"
)


# STEP 15: DISPLAY FINAL RESULTS

print("\n========== FINAL RESULTS ==========")

print("Best Performing Model:", best_model_name)

print("Best F1 Score:", round(best_score, 4))

print("\nBest Model Saved As: best_churn_model.pkl")

print("\nGenerated Files:")

print("1. model_comparison.csv")

print("2. model_comparison.png")

print("3. best_model_confusion_matrix.png")

print("4. best_churn_model.pkl")

print("\n========== MODEL TRAINING COMPLETED ==========")
