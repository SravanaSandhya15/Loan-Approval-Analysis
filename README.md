# Loan Approval Data Analysis

A Machine Learning project developed to analyze loan applicant data and predict whether a loan application will be approved or rejected using predictive modeling techniques.

##  Overview

The goal of this project is to automate and streamline the loan eligibility process. Using historical loan applicant records, the project handles missing data, performs Exploratory Data Analysis (EDA), and prepares the dataset for training binary classification algorithms.

##  Dataset Overview

The dataset contains **1,000 original entries** (950 non-null valid records) across **20 feature columns**:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Applicant_ID` | Numerical | Unique identification number for applicants |
| `Applicant_Income` | Numerical | Primary applicant's monthly/annual income |
| `Coapplicant_Income` | Numerical | Co-applicant's income |
| `Age` | Numerical | Age of the primary applicant |
| `Employment_Status` | Categorical | Employment state (Employed, Self-Employed, etc.) |
| `Marital_Status` | Categorical | Marital status (Married, Single) |
| `Dependents` | Numerical | Number of financial dependents |
| `Credit_Score` | Numerical | Credit score (Range: 550 – 799) |
| `Existing_Loans` | Numerical | Count of active ongoing loans |
| `DTI_Ratio` | Numerical | Debt-to-Income ratio |
| `Savings` | Numerical | Total savings balance |
| `Collateral_Value` | Numerical | Value of asset offered as collateral |
| `Loan_Amount` | Numerical | Requested loan principal amount |
| `Loan_Term` | Numerical | Loan duration (in months) |
| `Loan_Purpose` | Categorical | Purpose of the loan |
| `Property_Area` | Categorical | Area type (Urban, Semiurban, Rural) |
| `Education_Level` | Categorical | Educational qualification |
| `Gender` | Categorical | Applicant gender |
| `Employer_Category` | Categorical | Sector/Type of employer |
| **`Loan_Approved`** | **Target** | **Approval Status (`Yes` / `No`)** |


##  Tech Stack

* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn` 
* **Environment:** Jupyter Notebook


##  Project Workflow

### 1. Data Ingestion & Preprocessing
- Loaded raw dataset using `pandas`.
- Inspected missing value distribution across numerical and categorical fields.
- Applied `SimpleImputer` strategies:
  - **Numerical columns:** Mean Imputation
  - **Categorical columns:** Most Frequent (Mode) Imputation

### 2. Exploratory Data Analysis (EDA)
- Analyzed class distribution for the target variable (`Loan_Approved`).
- **Target Imbalance Ratio:**
  - **Rejected (`No`):** ~65.4% (621 records)
  - **Approved (`Yes`):** ~34.6% (379 records)

##  Key Findings & Analysis

- **Target Imbalance:** The class distribution shows a ~65:35 split between rejected and approved loans. Resampling techniques (such as SMOTE) or class-weighted models are recommended during training.
- **Credit Score & Income Impact:** Significant influence on approval probability based on Debt-to-Income (DTI) ratio and credit health.
