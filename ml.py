# ==========================================
# DATA SCIENCE LAB - ASSIGNMENT 2
# QUESTION 1
# EDA AND PREPROCESSING
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------

data = fetch_california_housing(as_frame=True)

df = data.frame.copy()

print("Dataset loaded successfully")
print()

# ------------------------------------------
# 2. Dataset Dimensions
# ------------------------------------------

print("Dataset Shape:")
print(df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# ------------------------------------------
# 3. Dataset Attributes
# ------------------------------------------

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# ------------------------------------------
# 4. First Five Records
# ------------------------------------------

print("\nFirst 5 Records:")
print(df.head())

# ------------------------------------------
# 5. Summary Statistics
# ------------------------------------------

print("\nSummary Statistics:")
print(df.describe())

# ------------------------------------------
# 6. Missing Values
# ------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())

# ------------------------------------------
# 7. Duplicate Records
# ------------------------------------------

print("\nNumber of Duplicate Records:")
print(df.duplicated().sum())

# Remove duplicates if present
df = df.drop_duplicates()

# ------------------------------------------
# 8. Handling Missing Values
# ------------------------------------------

# Fill numerical missing values with median
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values after treatment:")
print(df.isnull().sum())

# ------------------------------------------
# 9. Histograms
# ------------------------------------------

df.hist(figsize=(15, 12), bins=30)

plt.suptitle("Histograms of California Housing Features")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 10. Box Plots
# ------------------------------------------

plt.figure(figsize=(15, 8))

df.boxplot()

plt.title("Box Plots of Dataset Variables")
plt.xticks(rotation=45)
plt.ylabel("Value")
plt.show()

# ------------------------------------------
# 11. Scatter Plots
# ------------------------------------------

features = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup"
]

for feature in features:

    plt.figure(figsize=(7, 5))

    plt.scatter(df[feature], df["MedHouseVal"], alpha=0.3)

    plt.xlabel(feature)
    plt.ylabel("Median House Value")
    plt.title(feature + " vs Median House Value")

    plt.show()

# ------------------------------------------
# 12. Correlation Matrix
# ------------------------------------------

correlation = df.corr()

print("\nCorrelation Matrix:")
print(correlation)

# ------------------------------------------
# 13. Correlation Heatmap
# ------------------------------------------

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------
# 14. Correlation with Target
# ------------------------------------------

target_corr = correlation["MedHouseVal"].sort_values(
    ascending=False
)

print("\nCorrelation with Median House Value:")
print(target_corr)

# ------------------------------------------
# 15. Outlier Detection using IQR
# ------------------------------------------

print("\nOutlier Count using IQR:")

outlier_counts = {}

for col in df.columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((df[col] < lower) | (df[col] > upper)).sum()

    outlier_counts[col] = count

print(pd.Series(outlier_counts))

# ------------------------------------------
# 16. Treat Outliers using IQR Capping
# ------------------------------------------

# Only predictor variables are capped.
# Target variable is not modified.

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

for col in X.columns:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    X[col] = X[col].clip(lower, upper)

# ------------------------------------------
# 17. Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# ------------------------------------------
# 18. Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

print("\nPreprocessed Training Data:")
print(X_train_scaled[:5])

print("\nQ1 completed successfully.")

# ==========================================
# DATA SCIENCE LAB - ASSIGNMENT 2
# QUESTION 2
# MULTIPLE LINEAR REGRESSION
# ==========================================

import time
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------

data = fetch_california_housing(as_frame=True)

df = data.frame.copy()

# ------------------------------------------
# 2. Separate Features and Target
# ------------------------------------------

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# ------------------------------------------
# 3. Handle Missing Values
# ------------------------------------------

X = X.fillna(X.median())

# ------------------------------------------
# 4. Handle Outliers using IQR Capping
# ------------------------------------------

for col in X.columns:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    X[col] = X[col].clip(lower, upper)

# ------------------------------------------
# 5. Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------
# 6. Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# ------------------------------------------
# 7. Find Correlation with Target
# ------------------------------------------

correlations = df.corr()["MedHouseVal"].drop(
    "MedHouseVal"
).abs().sort_values(ascending=False)

print("Feature correlations with target:")
print(correlations)

# ------------------------------------------
# 8. Model A
# Three highest correlated features
# ------------------------------------------

features_A = correlations.head(3).index.tolist()

print("\nModel A Features:")
print(features_A)

# ------------------------------------------
# 9. Model B
# Five selected features
# ------------------------------------------

features_B = [
    "MedInc",
    "Latitude",
    "Longitude",
    "AveRooms",
    "HouseAge"
]

print("\nModel B Features:")
print(features_B)

# ------------------------------------------
# 10. Model C
# All Features
# ------------------------------------------

features_C = X.columns.tolist()

print("\nModel C Features:")
print(features_C)

# ------------------------------------------
# 11. Function to Train and Evaluate
# ------------------------------------------

def train_model(features):

    model = LinearRegression()

    start_time = time.perf_counter()

    model.fit(
        X_train_scaled[features],
        y_train
    )

    end_time = time.perf_counter()

    training_time = end_time - start_time

    predictions = model.predict(
        X_test_scaled[features]
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    return mae, mse, rmse, r2, training_time


# ------------------------------------------
# 12. Train Model A
# ------------------------------------------

result_A = train_model(features_A)

# ------------------------------------------
# 13. Train Model B
# ------------------------------------------

result_B = train_model(features_B)

# ------------------------------------------
# 14. Train Model C
# ------------------------------------------

result_C = train_model(features_C)

# ------------------------------------------
# 15. Comparative Table
# ------------------------------------------

results = pd.DataFrame({

    "Model": [
        "Model A",
        "Model B",
        "Model C"
    ],

    "Number of Features": [
        len(features_A),
        len(features_B),
        len(features_C)
    ],

    "Features": [
        ", ".join(features_A),
        ", ".join(features_B),
        ", ".join(features_C)
    ],

    "MAE": [
        result_A[0],
        result_B[0],
        result_C[0]
    ],

    "MSE": [
        result_A[1],
        result_B[1],
        result_C[1]
    ],

    "RMSE": [
        result_A[2],
        result_B[2],
        result_C[2]
    ],

    "R2 Score": [
        result_A[3],
        result_B[3],
        result_C[3]
    ],

    "Training Time (seconds)": [
        result_A[4],
        result_B[4],
        result_C[4]
    ]
})

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    results.to_string(index=False)
)

# ------------------------------------------
# 16. Identify Best Predictive Performance
# ------------------------------------------

best_model = results.loc[
    results["R2 Score"].idxmax()
]

print("\nModel with highest R2 Score:")
print(best_model["Model"])

print("\nModel with lowest RMSE:")
print(
    results.loc[
        results["RMSE"].idxmin(),
        "Model"
    ]
)

# ------------------------------------------
# 17. Prediction Time Comparison Plot
# ------------------------------------------

results.plot(
    x="Model",
    y="Training Time (seconds)",
    kind="bar",
    legend=False
)

plt.title("Training Time Comparison")
plt.ylabel("Training Time (seconds)")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 18. R2 Comparison
# ------------------------------------------

results.plot(
    x="Model",
    y="R2 Score",
    kind="bar",
    legend=False
)

plt.title("R2 Score Comparison")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.show()

print("\nQ2 completed successfully.")



# ==========================================
# DATA SCIENCE LAB - ASSIGNMENT 2
# QUESTION 3
# FEATURE SELECTION USING RFE
# ==========================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------

data = fetch_california_housing(as_frame=True)

df = data.frame.copy()

# ------------------------------------------
# 2. Separate Features and Target
# ------------------------------------------

X = df.drop("MedHouseVal", axis=1)

y = df["MedHouseVal"]

# ------------------------------------------
# 3. Handle Missing Values
# ------------------------------------------

X = X.fillna(X.median())

# ------------------------------------------
# 4. Outlier Treatment
# ------------------------------------------

for col in X.columns:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    X[col] = X[col].clip(
        lower,
        upper
    )

# ------------------------------------------
# 5. Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------
# 6. Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# ------------------------------------------
# 7. RFE Feature Selection
# ------------------------------------------

linear_model = LinearRegression()

rfe = RFE(
    estimator=linear_model,
    n_features_to_select=5
)

rfe.fit(
    X_train_scaled,
    y_train
)

# ------------------------------------------
# 8. Selected Features
# ------------------------------------------

selected_features = X_train_scaled.columns[
    rfe.support_
].tolist()

print("Selected Features using RFE:")
print(selected_features)

# ------------------------------------------
# 9. Feature Ranking
# ------------------------------------------

ranking = pd.DataFrame({

    "Feature": X_train_scaled.columns,

    "Ranking": rfe.ranking_,

    "Selected": rfe.support_
})

print("\nRFE Feature Ranking:")
print(
    ranking.sort_values(
        "Ranking"
    ).to_string(index=False)
)

# ------------------------------------------
# 10. Model using All Features
# ------------------------------------------

all_model = LinearRegression()

start_all = time.perf_counter()

all_model.fit(
    X_train_scaled,
    y_train
)

end_all = time.perf_counter()

all_training_time = end_all - start_all

all_predictions = all_model.predict(
    X_test_scaled
)

all_mae = mean_absolute_error(
    y_test,
    all_predictions
)

all_mse = mean_squared_error(
    y_test,
    all_predictions
)

all_rmse = np.sqrt(all_mse)

all_r2 = r2_score(
    y_test,
    all_predictions
)

# ------------------------------------------
# 11. Model using Selected Features
# ------------------------------------------

selected_model = LinearRegression()

start_selected = time.perf_counter()

selected_model.fit(
    X_train_scaled[selected_features],
    y_train
)

end_selected = time.perf_counter()

selected_training_time = (
    end_selected - start_selected
)

selected_predictions = selected_model.predict(
    X_test_scaled[selected_features]
)

selected_mae = mean_absolute_error(
    y_test,
    selected_predictions
)

selected_mse = mean_squared_error(
    y_test,
    selected_predictions
)

selected_rmse = np.sqrt(
    selected_mse
)

selected_r2 = r2_score(
    y_test,
    selected_predictions
)

# ------------------------------------------
# 12. Comparison Table
# ------------------------------------------

comparison = pd.DataFrame({

    "Model": [
        "All Features",
        "RFE Selected Features"
    ],

    "Number of Features": [
        X.shape[1],
        len(selected_features)
    ],

    "MAE": [
        all_mae,
        selected_mae
    ],

    "MSE": [
        all_mse,
        selected_mse
    ],

    "RMSE": [
        all_rmse,
        selected_rmse
    ],

    "R2 Score": [
        all_r2,
        selected_r2
    ],

    "Training Time (seconds)": [
        all_training_time,
        selected_training_time
    ]
})

print("\n==========================================")
print("FEATURE SELECTION MODEL COMPARISON")
print("==========================================")

print(
    comparison.to_string(index=False)
)

# ------------------------------------------
# 13. Regression Coefficients
# ------------------------------------------

coefficient_table = pd.DataFrame({

    "Feature": X_train_scaled.columns,

    "Coefficient": all_model.coef_

})

coefficient_table["Absolute Coefficient"] = (
    coefficient_table["Coefficient"].abs()
)

coefficient_table = coefficient_table.sort_values(
    "Absolute Coefficient",
    ascending=False
)

print("\nFeature Importance based on Regression Coefficients:")

print(
    coefficient_table.to_string(index=False)
)

# ------------------------------------------
# 14. R2 Comparison Plot
# ------------------------------------------

comparison.plot(
    x="Model",
    y="R2 Score",
    kind="bar",
    legend=False
)

plt.title("R2 Score: All Features vs RFE")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 15. RMSE Comparison Plot
# ------------------------------------------

comparison.plot(
    x="Model",
    y="RMSE",
    kind="bar",
    legend=False
)

plt.title("RMSE: All Features vs RFE")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()

print("\nQ3 completed successfully.")


# ================================================================
# DATA SCIENCE LAB (CS2311) - ASSIGNMENT 3
# COMPLETE SOLUTION
#
# Q1 : Telco Customer Churn Classification
# Q2 : PCA + SVD Dimensionality Reduction + Classification
# Q3 : House Sales Regression with Linear/Ridge/Lasso/ElasticNet
# ================================================================


# ================================================================
# INSTALL / IMPORT LIBRARIES
# ================================================================

import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

# Classification
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Dimensionality Reduction
from sklearn.decomposition import PCA, TruncatedSVD


# ================================================================
# FILE NAMES
# ================================================================
# Change these names according to your downloaded Kaggle files.
#
# Telco dataset:
# WA_Fn-UseC_-Telco-Customer-Churn.csv
#
# House dataset:
# kc_house_data.csv
#
# If your filenames are different, change them here.
# ================================================================

TELCO_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
HOUSE_FILE = "kc_house_data.csv"


# ================================================================
# HELPER FUNCTION FOR LOADING DATA
# ================================================================

def find_file(filename, keywords):

    if os.path.exists(filename):
        return filename

    # Search current directory
    for file in os.listdir("."):
        if file.lower().endswith(".csv"):
            lower = file.lower()

            for keyword in keywords:
                if keyword.lower() in lower:
                    return file

    return None


# ================================================================
# ================================================================
# QUESTION 1
# TELCO CUSTOMER CHURN CLASSIFICATION
# ================================================================
# ================================================================

print("\n")
print("=" * 80)
print("QUESTION 1 - TELCO CUSTOMER CHURN CLASSIFICATION")
print("=" * 80)


# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------

telco_path = find_file(
    TELCO_FILE,
    ["telco", "churn"]
)

if telco_path is None:

    print("\nTelco CSV was not found.")
    print("Please upload the Telco Customer Churn CSV file.")

    try:
        from google.colab import files

        uploaded = files.upload()

        telco_path = list(uploaded.keys())[0]

    except:
        raise FileNotFoundError(
            "Please place the Telco Customer Churn CSV in the working directory."
        )

print("\nUsing Telco file:", telco_path)

telco = pd.read_csv(telco_path)

print("\nDataset Shape:")
print(telco.shape)

print("\nFirst 5 Rows:")
print(telco.head())


# ------------------------------------------------
# 2. Basic Dataset Information
# ------------------------------------------------

print("\nDataset Information:")
print(telco.info())

print("\nSummary Statistics:")
print(telco.describe(include="all").T)


# ------------------------------------------------
# 3. Remove Customer ID
# ------------------------------------------------

if "customerID" in telco.columns:
    telco.drop("customerID", axis=1, inplace=True)


# ------------------------------------------------
# 4. Convert TotalCharges to Numeric
# ------------------------------------------------

if "TotalCharges" in telco.columns:

    telco["TotalCharges"] = pd.to_numeric(
        telco["TotalCharges"],
        errors="coerce"
    )


# ------------------------------------------------
# 5. Missing Values
# ------------------------------------------------

print("\nMissing Values:")
print(telco.isnull().sum())

# Fill numerical missing values
numeric_columns = telco.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:

    telco[col] = telco[col].fillna(
        telco[col].median()
    )

# Fill categorical missing values
categorical_columns = telco.select_dtypes(
    exclude=np.number
).columns

for col in categorical_columns:

    telco[col] = telco[col].fillna(
        telco[col].mode()[0]
    )


# ------------------------------------------------
# 6. Duplicate Records
# ------------------------------------------------

print("\nDuplicate Records:")
print(telco.duplicated().sum())

telco.drop_duplicates(inplace=True)


# ------------------------------------------------
# 7. Target Distribution
# ------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=telco,
    x="Churn"
)

plt.title("Distribution of Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()


# ------------------------------------------------
# 8. Important Feature Plots
# ------------------------------------------------

if "tenure" in telco.columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=telco,
        x="tenure",
        hue="Churn",
        kde=True,
        bins=30
    )

    plt.title("Tenure Distribution by Churn")
    plt.show()


if "MonthlyCharges" in telco.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=telco,
        x="Churn",
        y="MonthlyCharges"
    )

    plt.title("Monthly Charges vs Churn")
    plt.show()


if "Contract" in telco.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=telco,
        x="Contract",
        hue="Churn"
    )

    plt.title("Contract Type vs Churn")
    plt.xticks(rotation=20)
    plt.show()


# ------------------------------------------------
# 9. Encode Target Variable
# ------------------------------------------------

telco["Churn"] = telco["Churn"].map({
    "Yes": 1,
    "No": 0
})


# ------------------------------------------------
# 10. Separate Features and Target
# ------------------------------------------------

X = telco.drop("Churn", axis=1)

y = telco["Churn"]


# ------------------------------------------------
# 11. Identify Numerical and Categorical Features
# ------------------------------------------------

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ------------------------------------------------
# 12. Preprocessing
# ------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ------------------------------------------------
# 13. Train-Test Split
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# 14. Transform Data
# ------------------------------------------------

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print("\nProcessed Training Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Shape:")
print(X_test_processed.shape)


# ------------------------------------------------
# 15. Five Classification Algorithms
# ------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        )
}


# ------------------------------------------------
# 16. Train and Evaluate Models
# ------------------------------------------------

classification_results = []

roc_data = {}

pr_data = {}

confusion_matrices = {}


for name, model in models.items():

    print("\nTraining:", name)

    start_time = time.perf_counter()

    model.fit(
        X_train_processed,
        y_train
    )

    training_time = (
        time.perf_counter() -
        start_time
    )

    predictions = model.predict(
        X_test_processed
    )

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    classification_results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc,

        "Training Time (sec)": training_time
    })

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    confusion_matrices[name] = cm

    # ROC
    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_data[name] = (
        fpr,
        tpr,
        roc_auc
    )

    # Precision Recall
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    pr_data[name] = (
        recall_curve,
        precision_curve
    )


# ------------------------------------------------
# 17. Comparative Table
# ------------------------------------------------

classification_results = pd.DataFrame(
    classification_results
)

print("\n")
print("=" * 80)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 80)

print(
    classification_results.to_string(
        index=False
    )
)


# ------------------------------------------------
# 18. Confusion Matrices
# ------------------------------------------------

for name, cm in confusion_matrices.items():

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        "Confusion Matrix - " + name
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()


# ------------------------------------------------
# 19. ROC Curves
# ------------------------------------------------

plt.figure(figsize=(9, 7))

for name, data in roc_data.items():

    fpr, tpr, auc_value = data

    plt.plot(
        fpr,
        tpr,
        label=name +
        " (AUC = " +
        str(round(auc_value, 3)) +
        ")"
    )

plt.plot(
    [0, 1],
    [0, 1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves - Classification Models")

plt.legend()

plt.show()


# ------------------------------------------------
# 20. Precision-Recall Curves
# ------------------------------------------------

plt.figure(figsize=(9, 7))

for name, data in pr_data.items():

    recall_curve, precision_curve = data

    plt.plot(
        recall_curve,
        precision_curve,
        label=name
    )

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curves"
)

plt.legend()

plt.show()


# ------------------------------------------------
# 21. Best Classification Model
# ------------------------------------------------

best_model_name = classification_results.loc[
    classification_results["F1 Score"].idxmax(),
    "Model"
]

print(
    "\nModel with highest F1 Score:",
    best_model_name
)


# ================================================================
# ================================================================
# QUESTION 2
# PCA AND SVD DIMENSIONALITY REDUCTION
# ================================================================
# ================================================================

print("\n")
print("=" * 80)
print("QUESTION 2 - PCA AND SVD DIMENSIONALITY REDUCTION")
print("=" * 80)


# ------------------------------------------------
# 22. Original Number of Features
# ------------------------------------------------

original_features = X_train_processed.shape[1]

print(
    "\nOriginal Number of Features:",
    original_features
)


# ------------------------------------------------
# 23. PCA
# ------------------------------------------------

pca_start = time.perf_counter()

# Keep 95% of variance
pca = PCA(
    n_components=0.95,
    random_state=42
)

X_train_pca = pca.fit_transform(
    X_train_processed
)

X_test_pca = pca.transform(
    X_test_processed
)

pca_time = (
    time.perf_counter() -
    pca_start
)

print(
    "\nPCA Components:",
    pca.n_components_
)

print(
    "PCA Explained Variance:",
    round(
        pca.explained_variance_ratio_.sum(),
        4
    )
)

print(
    "PCA Time:",
    pca_time
)


# ------------------------------------------------
# 24. PCA Explained Variance Plot
# ------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    np.cumsum(
        pca.explained_variance_ratio_
    )
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")

plt.title(
    "PCA Cumulative Explained Variance"
)

plt.grid()

plt.show()


# ------------------------------------------------
# 25. SVD
# ------------------------------------------------

svd_start = time.perf_counter()

# Use same number of components as PCA
svd_components = pca.n_components_

svd = TruncatedSVD(
    n_components=svd_components,
    random_state=42
)

X_train_svd = svd.fit_transform(
    X_train_processed
)

X_test_svd = svd.transform(
    X_test_processed
)

svd_time = (
    time.perf_counter() -
    svd_start
)

print(
    "\nSVD Components:",
    svd_components
)

print(
    "SVD Explained Variance:",
    round(
        svd.explained_variance_ratio_.sum(),
        4
    )
)

print(
    "SVD Time:",
    svd_time
)


# ------------------------------------------------
# 26. Function for Reduced Data Classification
# ------------------------------------------------

def evaluate_reduced_models(
    Xtr,
    Xte,
    reduction_name
):

    results = []

    for name, model in models.items():

        start_time = time.perf_counter()

        model.fit(
            Xtr,
            y_train
        )

        training_time = (
            time.perf_counter() -
            start_time
        )

        predictions = model.predict(
            Xte
        )

        probabilities = model.predict_proba(
            Xte
        )[:, 1]

        results.append({

            "Reduction":
                reduction_name,

            "Model":
                name,

            "Features":
                Xtr.shape[1],

            "Accuracy":
                accuracy_score(
                    y_test,
                    predictions
                ),

            "Precision":
                precision_score(
                    y_test,
                    predictions
                ),

            "Recall":
                recall_score(
                    y_test,
                    predictions
                ),

            "F1 Score":
                f1_score(
                    y_test,
                    predictions
                ),

            "ROC-AUC":
                roc_auc_score(
                    y_test,
                    probabilities
                ),

            "Training Time (sec)":
                training_time
        })

    return pd.DataFrame(results)


# ------------------------------------------------
# 27. Original Data Results
# ------------------------------------------------

original_results = classification_results.copy()

original_results["Reduction"] = "Original"

original_results["Features"] = original_features

original_results.rename(
    columns={
        "Training Time (sec)":
            "Training Time (sec)"
    },
    inplace=True
)

original_results = original_results[
    [
        "Reduction",
        "Model",
        "Features",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Training Time (sec)"
    ]
]


# ------------------------------------------------
# 28. PCA Classification
# ------------------------------------------------

pca_results = evaluate_reduced_models(
    X_train_pca,
    X_test_pca,
    "PCA"
)


# ------------------------------------------------
# 29. SVD Classification
# ------------------------------------------------

svd_results = evaluate_reduced_models(
    X_train_svd,
    X_test_svd,
    "SVD"
)


# ------------------------------------------------
# 30. Combine All Results
# ------------------------------------------------

all_classification_results = pd.concat(
    [
        original_results,
        pca_results,
        svd_results
    ],
    ignore_index=True
)

print("\n")
print("=" * 80)
print("ORIGINAL vs PCA vs SVD")
print("=" * 80)

print(
    all_classification_results.to_string(
        index=False
    )
)


# ------------------------------------------------
# 31. F1 Score Comparison
# ------------------------------------------------

plt.figure(figsize=(12, 6))

sns.barplot(
    data=all_classification_results,
    x="Model",
    y="F1 Score",
    hue="Reduction"
)

plt.title(
    "F1 Score Comparison - Original vs PCA vs SVD"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# ------------------------------------------------
# 32. Accuracy Comparison
# ------------------------------------------------

plt.figure(figsize=(12, 6))

sns.barplot(
    data=all_classification_results,
    x="Model",
    y="Accuracy",
    hue="Reduction"
)

plt.title(
    "Accuracy Comparison - Original vs PCA vs SVD"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# ------------------------------------------------
# 33. ROC-AUC Comparison
# ------------------------------------------------

plt.figure(figsize=(12, 6))

sns.barplot(
    data=all_classification_results,
    x="Model",
    y="ROC-AUC",
    hue="Reduction"
)

plt.title(
    "ROC-AUC Comparison - Original vs PCA vs SVD"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# ------------------------------------------------
# 34. Feature Count Comparison
# ------------------------------------------------

feature_comparison = pd.DataFrame({

    "Dataset":
        [
            "Original",
            "PCA",
            "SVD"
        ],

    "Number of Features":
        [
            original_features,
            X_train_pca.shape[1],
            X_train_svd.shape[1]
        ]
})

print("\nFeature Count Comparison:")
print(
    feature_comparison.to_string(
        index=False
    )
)

plt.figure(figsize=(7, 5))

sns.barplot(
    data=feature_comparison,
    x="Dataset",
    y="Number of Features"
)

plt.title(
    "Number of Features Before and After Reduction"
)

plt.show()


# ------------------------------------------------
# 35. Computational Time Comparison
# ------------------------------------------------

time_comparison = all_classification_results.groupby(
    "Reduction"
)["Training Time (sec)"].mean().reset_index()

plt.figure(figsize=(8, 5))

sns.barplot(
    data=time_comparison,
    x="Reduction",
    y="Training Time (sec)"
)

plt.title(
    "Average Classification Training Time"
)

plt.show()


# ================================================================
# ================================================================
# QUESTION 3
# HOUSE SALES PREDICTION
# ================================================================
# ================================================================

print("\n")
print("=" * 80)
print("QUESTION 3 - HOUSE SALES REGRESSION")
print("=" * 80)


# ------------------------------------------------
# 36. Load House Dataset
# ------------------------------------------------

house_path = find_file(
    HOUSE_FILE,
    [
        "kc_house",
        "house",
        "sales"
    ]
)

if house_path is None:

    print("\nHouse Sales CSV was not found.")
    print("Please upload the House Sales CSV file.")

    try:

        from google.colab import files

        uploaded = files.upload()

        house_path = list(
            uploaded.keys()
        )[0]

    except:

        raise FileNotFoundError(
            "Please place the House Sales CSV in the working directory."
        )


print(
    "\nUsing House file:",
    house_path
)

house = pd.read_csv(
    house_path
)

print("\nHouse Dataset Shape:")
print(house.shape)

print("\nFirst 5 Rows:")
print(house.head())


# ------------------------------------------------
# 37. Identify Target Column
# ------------------------------------------------

possible_targets = [
    "price",
    "SalePrice",
    "sale_price",
    "Price"
]

target_column = None

for col in possible_targets:

    if col in house.columns:

        target_column = col
        break


if target_column is None:

    raise ValueError(
        "Could not find the house price target column. "
        "Rename the target column to 'price'."
    )


print(
    "\nTarget Column:",
    target_column
)


# ------------------------------------------------
# 38. Remove ID Columns
# ------------------------------------------------

id_columns = []

for col in house.columns:

    if col.lower() in [
        "id",
        "unnamed: 0"
    ]:

        id_columns.append(col)


if len(id_columns) > 0:

    house.drop(
        id_columns,
        axis=1,
        inplace=True
    )


# ------------------------------------------------
# 39. Remove Date Column
# ------------------------------------------------

date_columns = []

for col in house.columns:

    if col.lower() in [
        "date",
        "sale_date"
    ]:

        date_columns.append(col)


for col in date_columns:

    house.drop(
        col,
        axis=1,
        inplace=True
    )


# ------------------------------------------------
# 40. Duplicate Records
# ------------------------------------------------

print(
    "\nDuplicate House Records:",
    house.duplicated().sum()
)

house.drop_duplicates(
    inplace=True
)


# ------------------------------------------------
# 41. Separate X and y
# ------------------------------------------------

X_house = house.drop(
    target_column,
    axis=1
)

y_house = house[
    target_column
]


# ------------------------------------------------
# 42. Missing Values
# ------------------------------------------------

print("\nMissing Values:")
print(
    house.isnull().sum()
)


# ------------------------------------------------
# 43. Identify Numeric and Categorical Columns
# ------------------------------------------------

house_numeric = X_house.select_dtypes(
    include=np.number
).columns.tolist()

house_categorical = X_house.select_dtypes(
    exclude=np.number
).columns.tolist()


print(
    "\nNumerical House Features:"
)

print(
    house_numeric
)

print(
    "\nCategorical House Features:"
)

print(
    house_categorical
)


# ------------------------------------------------
# 44. House Price Distribution
# ------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    y_house,
    bins=50,
    kde=True
)

plt.title(
    "Distribution of House Sale Prices"
)

plt.xlabel(
    "Sale Price"
)

plt.show()


# ------------------------------------------------
# 45. Important House Feature Plots
# ------------------------------------------------

for feature in [
    "sqft_living",
    "grade",
    "bedrooms"
]:

    if feature in house.columns:

        plt.figure(figsize=(8, 5))

        plt.scatter(
            house[feature],
            y_house,
            alpha=0.3
        )

        plt.xlabel(feature)
        plt.ylabel("Sale Price")

        plt.title(
            feature +
            " vs Sale Price"
        )

        plt.show()


# ------------------------------------------------
# 46. Preprocessing
# ------------------------------------------------

house_numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

house_categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


house_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            house_numeric_transformer,
            house_numeric
        ),
        (
            "cat",
            house_categorical_transformer,
            house_categorical
        )
    ]
)


# ------------------------------------------------
# 47. Train-Test Split
# ------------------------------------------------

X_house_train, X_house_test, y_house_train, y_house_test = train_test_split(
    X_house,
    y_house,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------
# 48. Preprocess House Data
# ------------------------------------------------

X_house_train_processed = house_preprocessor.fit_transform(
    X_house_train
)

X_house_test_processed = house_preprocessor.transform(
    X_house_test
)


print(
    "\nProcessed House Training Shape:"
)

print(
    X_house_train_processed.shape
)


# ------------------------------------------------
# 49. Function to Evaluate Regression
# ------------------------------------------------

def evaluate_regression(
    model,
    name
):

    start_time = time.perf_counter()

    model.fit(
        X_house_train_processed,
        y_house_train
    )

    training_time = (
        time.perf_counter() -
        start_time
    )

    train_prediction = model.predict(
        X_house_train_processed
    )

    test_prediction = model.predict(
        X_house_test_processed
    )

    train_mse = mean_squared_error(
        y_house_train,
        train_prediction
    )

    test_mse = mean_squared_error(
        y_house_test,
        test_prediction
    )

    train_rmse = np.sqrt(
        train_mse
    )

    test_rmse = np.sqrt(
        test_mse
    )

    train_mae = mean_absolute_error(
        y_house_train,
        train_prediction
    )

    test_mae = mean_absolute_error(
        y_house_test,
        test_prediction
    )

    train_r2 = r2_score(
        y_house_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_house_test,
        test_prediction
    )

    return {

        "Model": name,

        "Train MSE":
            train_mse,

        "Test MSE":
            test_mse,

        "Train RMSE":
            train_rmse,

        "Test RMSE":
            test_rmse,

        "Train MAE":
            train_mae,

        "Test MAE":
            test_mae,

        "Train R2":
            train_r2,

        "Test R2":
            test_r2,

        "Training Time (sec)":
            training_time
    }


# ------------------------------------------------
# 50. Multiple Linear Regression
# ------------------------------------------------

linear_model = LinearRegression()

linear_result = evaluate_regression(
    linear_model,
    "Linear Regression"
)


# ================================================================
# 51. RIDGE REGRESSION
# ================================================================

ridge_lambdas = [
    0.01,
    0.1,
    1,
    10,
    100
]

ridge_results = []

ridge_models = {}

for alpha in ridge_lambdas:

    model = Ridge(
        alpha=alpha
    )

    result = evaluate_regression(
        model,
        "Ridge (lambda=" +
        str(alpha) +
        ")"
    )

    result["Lambda"] = alpha

    ridge_results.append(
        result
    )

    ridge_models[alpha] = model


# ================================================================
# 52. LASSO REGRESSION
# ================================================================

lasso_lambdas = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1
]

lasso_results = []

lasso_models = {}

for alpha in lasso_lambdas:

    model = Lasso(
        alpha=alpha,
        max_iter=10000
    )

    result = evaluate_regression(
        model,
        "Lasso (lambda=" +
        str(alpha) +
        ")"
    )

    result["Lambda"] = alpha

    lasso_results.append(
        result
    )

    lasso_models[alpha] = model


# ================================================================
# 53. ELASTIC NET REGRESSION
# ================================================================

elastic_lambdas = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1
]

elastic_results = []

elastic_models = {}

for alpha in elastic_lambdas:

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=0.5,
        max_iter=10000
    )

    result = evaluate_regression(
        model,
        "Elastic Net (lambda=" +
        str(alpha) +
        ")"
    )

    result["Lambda"] = alpha

    elastic_results.append(
        result
    )

    elastic_models[alpha] = model


# ------------------------------------------------
# 54. Combine Regression Results
# ------------------------------------------------

linear_result["Lambda"] = 0

regression_results = pd.DataFrame(
    [linear_result]
    + ridge_results
    + lasso_results
    + elastic_results
)


print("\n")
print("=" * 80)
print("REGRESSION MODEL COMPARISON")
print("=" * 80)

print(
    regression_results[
        [
            "Model",
            "Lambda",
            "Train MSE",
            "Test MSE",
            "Train RMSE",
            "Test RMSE",
            "Train MAE",
            "Test MAE",
            "Train R2",
            "Test R2",
            "Training Time (sec)"
        ]
    ].to_string(index=False)
)


# ================================================================
# 55. RIDGE ERROR vs LAMBDA
# ================================================================

ridge_df = pd.DataFrame(
    ridge_results
)

plt.figure(figsize=(9, 6))

plt.plot(
    ridge_df["Lambda"],
    ridge_df["Train RMSE"],
    marker="o",
    label="Training RMSE"
)

plt.plot(
    ridge_df["Lambda"],
    ridge_df["Test RMSE"],
    marker="o",
    label="Testing RMSE"
)

plt.xscale("log")

plt.xlabel("Lambda")
plt.ylabel("RMSE")

plt.title(
    "Ridge Regression: RMSE vs Lambda"
)

plt.legend()

plt.grid()

plt.show()


# ================================================================
# 56. LASSO ERROR vs LAMBDA
# ================================================================

lasso_df = pd.DataFrame(
    lasso_results
)

plt.figure(figsize=(9, 6))

plt.plot(
    lasso_df["Lambda"],
    lasso_df["Train RMSE"],
    marker="o",
    label="Training RMSE"
)

plt.plot(
    lasso_df["Lambda"],
    lasso_df["Test RMSE"],
    marker="o",
    label="Testing RMSE"
)

plt.xscale("log")

plt.xlabel("Lambda")
plt.ylabel("RMSE")

plt.title(
    "Lasso Regression: RMSE vs Lambda"
)

plt.legend()

plt.grid()

plt.show()


# ================================================================
# 57. ELASTIC NET ERROR vs LAMBDA
# ================================================================

elastic_df = pd.DataFrame(
    elastic_results
)

plt.figure(figsize=(9, 6))

plt.plot(
    elastic_df["Lambda"],
    elastic_df["Train RMSE"],
    marker="o",
    label="Training RMSE"
)

plt.plot(
    elastic_df["Lambda"],
    elastic_df["Test RMSE"],
    marker="o",
    label="Testing RMSE"
)

plt.xscale("log")

plt.xlabel("Lambda")
plt.ylabel("RMSE")

plt.title(
    "Elastic Net Regression: RMSE vs Lambda"
)

plt.legend()

plt.grid()

plt.show()


# ================================================================
# 58. BEST RIDGE / LASSO / ELASTIC NET
# ================================================================

best_ridge_alpha = ridge_df.loc[
    ridge_df["Test RMSE"].idxmin(),
    "Lambda"
]

best_lasso_alpha = lasso_df.loc[
    lasso_df["Test RMSE"].idxmin(),
    "Lambda"
]

best_elastic_alpha = elastic_df.loc[
    elastic_df["Test RMSE"].idxmin(),
    "Lambda"
]


print(
    "\nBest Ridge Lambda:",
    best_ridge_alpha
)

print(
    "Best Lasso Lambda:",
    best_lasso_alpha
)

print(
    "Best Elastic Net Lambda:",
    best_elastic_alpha
)


# ================================================================
# 59. REGRESSION COEFFICIENT COMPARISON
# ================================================================

linear_coefficients = linear_model.coef_

ridge_best_model = ridge_models[
    best_ridge_alpha
]

lasso_best_model = lasso_models[
    best_lasso_alpha
]

elastic_best_model = elastic_models[
    best_elastic_alpha
]


elastic_coefficients = (
    elastic_best_model.coef_
)

coefficient_comparison = pd.DataFrame({

    "Linear":
        linear_coefficients,

    "Ridge":
        ridge_best_model.coef_,

    "Lasso":
        lasso_best_model.coef_,

    "Elastic Net":
        elastic_coefficients
})


# ------------------------------------------------
# 60. Coefficient Plot
# ------------------------------------------------

# Limit plot to first 30 features if one-hot
# encoding creates many features.

number_to_plot = min(
    30,
    len(coefficient_comparison)
)

coefficient_comparison_plot = (
    coefficient_comparison.iloc[
        :number_to_plot
    ]
)

coefficient_comparison_plot.index = [
    "Feature " + str(i + 1)
    for i in range(number_to_plot)
]


coefficient_comparison_plot.plot(
    kind="bar",
    figsize=(15, 7)
)

plt.title(
    "Regression Coefficient Comparison"
)

plt.xlabel("Features")

plt.ylabel("Coefficient Value")

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.show()


# ================================================================
# 61. Absolute Coefficient Comparison
# ================================================================

absolute_coefficients = (
    coefficient_comparison.abs()
)

absolute_coefficients.iloc[
    :number_to_plot
].plot(
    kind="bar",
    figsize=(15, 7)
)

plt.title(
    "Absolute Regression Coefficient Comparison"
)

plt.xlabel("Features")

plt.ylabel(
    "Absolute Coefficient Value"
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.show()


# ================================================================
# 62. TEST RMSE COMPARISON
# ================================================================

plt.figure(figsize=(14, 7))

sns.barplot(
    data=regression_results,
    x="Model",
    y="Test RMSE"
)

plt.title(
    "Test RMSE Comparison of Regression Models"
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.tight_layout()

plt.show()


# ================================================================
# 63. TEST R2 COMPARISON
# ================================================================

plt.figure(figsize=(14, 7))

sns.barplot(
    data=regression_results,
    x="Model",
    y="Test R2"
)

plt.title(
    "Test R2 Comparison of Regression Models"
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.tight_layout()

plt.show()


# ================================================================
# 64. BEST OVERALL REGRESSION MODEL
# ================================================================

best_regression = regression_results.loc[
    regression_results["Test RMSE"].idxmin()
]

print("\n")
print("=" * 80)
print("BEST REGRESSION MODEL")
print("=" * 80)

print(
    best_regression.to_string()
)


# ================================================================
# 65. FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 80)
print("FINAL ASSIGNMENT SUMMARY")
print("=" * 80)

print(
    "\nQ1 - Classification:"
)

print(
    classification_results[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    ].to_string(index=False)
)


print(
    "\nQ2 - Dimensionality Reduction:"
)

print(
    all_classification_results[
        [
            "Reduction",
            "Model",
            "Features",
            "Accuracy",
            "F1 Score",
            "ROC-AUC"
        ]
    ].to_string(index=False)
)


print(
    "\nQ3 - Regression:"
)

print(
    regression_results[
        [
            "Model",
            "Lambda",
            "Test MSE",
            "Test RMSE",
            "Test MAE",
            "Test R2"
        ]
    ].to_string(index=False)
)


print("\n")
print("=" * 80)
print("ASSIGNMENT 3 COMPLETED")
print("=" * 80)