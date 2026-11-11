# Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a bank customer is likely to leave the bank or continue using its services.

The project uses customer demographic and account-related information to identify potential customer churn. It helps businesses understand customer behaviour and take preventive measures to improve customer retention.

An interactive web application is developed using **Streamlit**, allowing users to enter customer details and receive churn predictions.

## 🎯 Objectives

- Analyze customer data to identify churn patterns.
- Perform Exploratory Data Analysis (EDA).
- Apply data preprocessing techniques.
- Train and compare multiple Machine Learning algorithms.
- Evaluate model performance using classification metrics.
- Develop an interactive dashboard for customer churn prediction.

## 📂 Dataset Description

**Dataset:** Churn_Modelling.csv

The dataset contains 10,000 customer records with demographic and banking-related information.

**Target Variable: `Exited`**

- `0` – Customer stayed with the bank.
- `1` – Customer exited the bank.

### Features Used

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Customer's geographical location |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Number of years with the bank |
| Balance | Customer's account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Customer's estimated salary |

The columns `RowNumber`, `CustomerId`, and `Surname` are removed during preprocessing.

## ⚙️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn
- **Web Framework:** Streamlit
- **Model Serialization:** Pickle

## 🤖 Machine Learning Algorithms

The following classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier

## 🔄 Project Workflow

1. **Data Collection:** Load the Churn_Modelling.csv dataset.
2. **Data Preprocessing:** Remove unnecessary columns and prepare input and target variables.
3. **Exploratory Data Analysis:** Analyze customer characteristics and churn patterns using visualizations.
4. **Feature Engineering:** Apply StandardScaler to numerical features and OneHotEncoder to categorical features.
5. **Model Training:** Train multiple classification algorithms using an 80:20 train-test split.
6. **Model Evaluation:** Compare models using accuracy, precision, recall, F1-score, and ROC-AUC.
7. **Model Saving:** Save the selected trained model as `best_churn_model.pkl`.
8. **Deployment:** Develop an interactive Streamlit application for customer churn prediction.

## 📊 Model Evaluation

The models are evaluated using the following performance metrics:

| Metric | Description |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures how many predicted churn cases are actually churn cases |
| Recall | Measures how many actual churn cases are correctly identified |
| F1-Score | Balances precision and recall |
| ROC-AUC | Measures the model's ability to distinguish between classes |

The model comparison results are saved in `model_comparison.csv`.

The confusion matrix of the selected model is saved as `best_model_confusion_matrix.png`.

## 🖥️ Streamlit Application

The project includes an interactive web dashboard developed using Streamlit.

### Key Features

- Customer churn overview and key statistics.
- Interactive data visualizations.
- Churn analysis based on geography and age groups.
- Customer details input form.
- Individual customer churn prediction.
- Prediction results displayed through an interactive dashboard.

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── dataset/
│   └── Churn_Modelling.csv
│
├── app.py
├── train.py
├── model_training.py
├── eda.py
├── visualization.py
│
├── best_churn_model.pkl
├── model_comparison.csv
├── model_comparison.png
├── best_model_confusion_matrix.png
│
└── README.md
```

## 🚀 Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/aryanalavade14/CODESOFT_TASK03.git
```

### Step 2: Navigate to the Project Folder

```bash
cd CODESOFT_TASK03
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly joblib
```

### Step 4: Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your default browser. If it does not open automatically, use the local URL displayed in the terminal.

## 🔮 Future Scope

- Improve prediction performance through hyperparameter tuning.
- Implement advanced ensemble learning techniques.
- Integrate explainable AI techniques such as SHAP.
- Deploy the application on a cloud platform.
- Provide personalized customer retention recommendations.

## 👩‍💻 Author

**Arya Nalavade**

Electronics & Computer Science Engineering Student

## 🔗 GitHub Repository

[Customer Churn Prediction Using Machine Learning](https://github.com/aryanalavade14/CODESOFT_TASK03)
