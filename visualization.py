import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/Churn_Modelling.csv")

sns.set_style("whitegrid")

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Exited")

plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = Stayed, 1 = Left)")
plt.ylabel("Number of Customers")

plt.show()
#churn based on age 
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    hue="Exited",
    multiple="stack",
    bins=30
)

plt.title("Customer Churn Based on Age")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.show()

#churn based on geography
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Geography", hue="Exited")

plt.title("Customer Churn Based on Geography")
plt.xlabel("Country")
plt.ylabel("Number of Customers")

plt.show()

#churn based on gender
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Gender", hue="Exited")

plt.title("Customer Churn Based on Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.show()

#churn based on active membership
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="IsActiveMember", hue="Exited")

plt.title("Churn Based on Active Membership")
plt.xlabel("Active Member (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.show()

#correlation heatmap
plt.figure(figsize=(12, 7))

numerical_df = df.select_dtypes(include=["int64", "float64"])

sns.heatmap(
    numerical_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

#Box Plot – Age vs Churn
plt.figure(figsize=(7, 5))

sns.boxplot(data=df, x="Exited", y="Age")

plt.title("Age Distribution vs Customer Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Left)")
plt.ylabel("Age")

plt.show()

#Pie Chart – Customer Churn Percentage
churn_counts = df["Exited"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    churn_counts,
    labels=["Stayed", "Exited"],
    autopct="%1.2f%%",
    startangle=90
)

plt.title("Customer Churn Percentage")

plt.show()

#Scatter Plot – Age vs Balance
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Age",
    y="Balance",
    hue="Exited",
    alpha=0.6
)

plt.title("Age vs Balance Based on Churn")
plt.xlabel("Age")
plt.ylabel("Account Balance")

plt.show()
