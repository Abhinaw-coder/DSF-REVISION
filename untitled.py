f1

# INSTALL FIRST IN CMD:
# pip install beautifulsoup4 requests pandas matplotlib seaborn scikit-learn

# ============================================================
# COMPLETE END-TO-END WEB SCRAPING + DATA SCIENCE PIPELINE
# ============================================================

from bs4 import BeautifulSoup
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# 1. READ HTML FILE
with open("HTML_Reference_Sample_Campus_Events.html",
          "r", encoding="utf-8") as file:
    html = file.read()


# 2. CREATE BEAUTIFULSOUP OBJECT
soup = BeautifulSoup(html, "html.parser")


# 3. FIND REPEATING RECORDS
events = soup.find_all("div", class_="event")

print("Number of events:", len(events))


# 4. SCRAPE DATA
data = []

for event in events:

    name_tag = event.find("h2")
    name = name_tag.text.strip() if name_tag else None

    venue_tag = event.find("p", class_="venue")
    venue = venue_tag.text.strip() if venue_tag else None

    date_tag = event.find("p", class_="date")
    date = date_tag.text.strip() if date_tag else None

    category_tag = event.find("span", class_="category")
    category = category_tag.text.strip() if category_tag else None

    fee_tag = event.find("span", class_="fee")
    fee = fee_tag.text.strip() if fee_tag else None

    seats_tag = event.find("p", class_="seats")
    seats = seats_tag.text.strip() if seats_tag else None

    record = {
        "name": name,
        "venue": venue,
        "date": date,
        "category": category,
        "fee": fee,
        "seats": seats
    }

    data.append(record)


# 5. SAVE SCRAPED DATA AS JSON
with open("events.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)


# 6. JSON → DATAFRAME
df = pd.DataFrame(data)

print("\nOriginal Data:")
print(df)


# 7. DATA CLEANING

# Remove duplicate rows
df = df.drop_duplicates()

# Clean text
df["name"] = df["name"].str.strip()
df["category"] = df["category"].str.strip()

# Handle missing venue
df["venue"] = df["venue"].fillna("Not Specified")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Clean fee
df["fee"] = df["fee"].replace("Free", "0")
df["fee"] = df["fee"].str.replace("₹", "", regex=False)
df["fee"] = pd.to_numeric(df["fee"])

# Clean seats
df["seats"] = df["seats"].str.replace("Seats:", "", regex=False)
df["seats"] = pd.to_numeric(df["seats"])


# 8. FEATURE ENGINEERING

# Feature 1
df["name_length"] = df["name"].str.len()

# Feature 2
df["month"] = df["date"].dt.month

# Feature 3
df["day"] = df["date"].dt.day

# Feature 4
df["payment_type"] = "Paid"
df.loc[df["fee"] == 0, "payment_type"] = "Free"

# Feature 5
df["seat_category"] = "Low"
df.loc[df["seats"] >= 50, "seat_category"] = "Medium"
df.loc[df["seats"] >= 100, "seat_category"] = "High"


# 9. MIN-MAX NORMALIZATION

scaler = MinMaxScaler()

df[["fee_normalized", "seats_normalized"]] = scaler.fit_transform(
    df[["fee", "seats"]]
)


# 10. ANALYSIS

print("\nCategory Counts:")
print(df["category"].value_counts())

print("\nAverage Fee:")
print(df["fee"].mean())

print("\nAverage Seats:")
print(df["seats"].mean())

print("\nAverage Fee by Category:")
print(df.groupby("category")["fee"].mean())

print("\nHighest Fee Event:")
print(df.loc[df["fee"].idxmax()])

print("\nHighest Capacity Event:")
print(df.loc[df["seats"].idxmax()])


# 11. VISUALIZATION 1
plt.figure(figsize=(8, 5))
sns.countplot(x="category", data=df)
plt.title("Number of Events by Category")
plt.xlabel("Category")
plt.ylabel("Number of Events")
plt.xticks(rotation=45)
plt.show()


# 12. VISUALIZATION 2
plt.figure(figsize=(8, 5))
sns.histplot(df["fee"], bins=5)
plt.title("Distribution of Event Fees")
plt.xlabel("Fee")
plt.ylabel("Number of Events")
plt.show()


# 13. VISUALIZATION 3
plt.figure(figsize=(8, 5))
sns.histplot(df["seats"], bins=5)
plt.title("Distribution of Available Seats")
plt.xlabel("Seats")
plt.ylabel("Number of Events")
plt.show()


# 14. VISUALIZATION 4
plt.figure(figsize=(8, 5))
sns.scatterplot(x="fee", y="seats", data=df)
plt.title("Fee vs Available Seats")
plt.xlabel("Fee")
plt.ylabel("Seats")
plt.show()


# 15. VISUALIZATION 5
plt.figure(figsize=(8, 5))
sns.barplot(x="category", y="fee", data=df)
plt.title("Average Fee by Category")
plt.xlabel("Category")
plt.ylabel("Average Fee")
plt.xticks(rotation=45)
plt.show()


# 16. VISUALIZATION 6
plt.figure(figsize=(8, 5))
sns.boxplot(x="category", y="seats", data=df)
plt.title("Seats Distribution by Category")
plt.xlabel("Category")
plt.ylabel("Seats")
plt.xticks(rotation=45)
plt.show()


# 17. SAVE FINAL CSV
df.to_csv("final_events.csv", index=False)

print("\nFinal Data:")
print(df)

print("\nCSV saved successfully!")



f2



from scipy.stats import ttest_ind

group_A = [65, 70, 68, 72, 66, 75, 69, 71, 67, 70]
group_B = [72, 75, 78, 74, 80, 77, 73, 79, 76, 81]

t_stat, p_value = ttest_ind(group_A, group_B)

print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Statistically significant difference")
else:
    print("No statistically significant difference")


from scipy.stats import ttest_rel

before = [52, 48, 65, 60, 55, 70, 58, 62, 50, 67]
after = [60, 55, 70, 66, 63, 75, 64, 68, 57, 72]

t_stat, p_value = ttest_rel(before, after)

print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Statistically significant change")
else:
    print("No statistically significant change")


f3


# ============================================================
# 18. SIMPLE LINEAR REGRESSION
# Predict SEATS using FEE
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# X = independent variable
# y = dependent variable

X = df[["fee"]]
y = df["seats"]


# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create regression model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("\n================ LINEAR REGRESSION ================\n")

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nMAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ============================================================
# 19. REGRESSION VISUALIZATION
# Actual points + Regression Line
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test["fee"],
    y_test,
    label="Actual"
)

plt.plot(
    X_test["fee"],
    y_pred,
    label="Regression Line"
)

plt.xlabel("Fee")
plt.ylabel("Seats")
plt.title("Linear Regression: Fee vs Seats")
plt.legend()

plt.show()


# ============================================================
# 20. ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Seats")
plt.ylabel("Predicted Seats")
plt.title("Actual vs Predicted Seats")

# Ideal prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()


# ============================================================
# 21. RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Seats")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.show()


f4


# ============================================================
# MULTIPLE LINEAR REGRESSION
# Predict SEATS using multiple variables
# ============================================================

X = df[["fee", "name_length", "month"]]
y = df["seats"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMultiple Linear Regression")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Actual vs Predicted
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Seats")
plt.ylabel("Predicted Seats")
plt.title("Multiple Linear Regression - Actual vs Predicted")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()


f5
# ============================================================
# BASIC VISUALIZATION IMPORTS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# ============================================================
# 1. LINE PLOT
# Use: trend / change over ordered values
# ============================================================

plt.plot(df["date"], df["sales"])

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Trend")

plt.show()


# ============================================================
# 2. BAR PLOT
# Use: compare categories
# ============================================================

plt.bar(df["category"], df["sales"])

plt.xlabel("Category")
plt.ylabel("Sales")
plt.title("Sales by Category")

plt.show()


# ============================================================
# 3. HORIZONTAL BAR PLOT
# ============================================================

plt.barh(df["category"], df["sales"])

plt.xlabel("Sales")
plt.ylabel("Category")
plt.title("Sales by Category")

plt.show()


# ============================================================
# 4. HISTOGRAM
# Use: distribution of a numerical variable
# ============================================================

plt.hist(df["age"], bins=10)

plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")

plt.show()


# ============================================================
# 5. SCATTER PLOT
# Use: relationship between two numerical variables
# ============================================================

plt.scatter(
    df["income"],
    df["price"]
)

plt.xlabel("Income")
plt.ylabel("Price")
plt.title("Income vs Price")

plt.show()


# ============================================================
# 6. BOX PLOT
# Use: distribution + outliers
# ============================================================

plt.boxplot(df["price"])

plt.ylabel("Price")
plt.title("Price Distribution")

plt.show()


# ============================================================
# 7. MULTIPLE BOX PLOTS
# ============================================================

df[["price", "income", "age"]].boxplot(
    figsize=(8, 5)
)

plt.title("Box Plot")
plt.show()


# ============================================================
# 8. PIE CHART
# Use: percentage / proportion of categories
# ============================================================

counts = df["category"].value_counts()

plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%"
)

plt.title("Category Distribution")

plt.show()


# ============================================================
# 9. SEABORN COUNT PLOT
# Use: count of observations in categories
# ============================================================

sns.countplot(
    x="category",
    data=df
)

plt.title("Number of Records by Category")
plt.xticks(rotation=45)

plt.show()


# ============================================================
# 10. SEABORN HISTOGRAM
# ============================================================

sns.histplot(
    df["price"],
    bins=10,
    kde=True
)

plt.title("Price Distribution")

plt.show()


# ============================================================
# 11. KDE / DISTRIBUTION PLOT
# Use: smooth distribution
# ============================================================

sns.kdeplot(
    df["price"],
    fill=True
)

plt.title("Price Distribution")

plt.show()


# ============================================================
# 12. REGRESSION PLOT
# Use: relationship + fitted regression line
# ============================================================

sns.regplot(
    x="income",
    y="price",
    data=df
)

plt.title("Income vs Price")

plt.show()


# ============================================================
# 13. HEATMAP
# Use: correlation matrix
# ============================================================

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# ============================================================
# 14. GROUPED BAR PLOT
# ============================================================

sns.barplot(
    x="category",
    y="price",
    data=df
)

plt.title("Average Price by Category")

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 15. BOX PLOT BY CATEGORY
# ============================================================

sns.boxplot(
    x="category",
    y="price",
    data=df
)

plt.title("Price Distribution by Category")

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 16. VIOLIN PLOT
# Use: distribution + density + comparison
# ============================================================

sns.violinplot(
    x="category",
    y="price",
    data=df
)

plt.title("Price Distribution by Category")

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 17. SWARM PLOT
# Use: individual observations by category
# ============================================================

sns.swarmplot(
    x="category",
    y="price",
    data=df
)

plt.title("Price by Category")

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 18. STRIP PLOT
# Use: individual observations
# ============================================================

sns.stripplot(
    x="category",
    y="price",
    data=df
)

plt.title("Price by Category")

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 19. PAIR PLOT
# Use: relationships among multiple numerical variables
# ============================================================

sns.pairplot(
    df[["income", "age", "rooms", "price"]]
)

plt.show()


# ============================================================
# 20. REGRESSION RESIDUAL PLOT
# ============================================================

sns.residplot(
    x="income",
    y="price",
    data=df
)

plt.title("Residual Plot")

plt.show()


# ============================================================
# 21. ACTUAL VS PREDICTED
# Use: regression model evaluation
# ============================================================

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()


# ============================================================
# 22. RESIDUAL SCATTER PLOT
# ============================================================

residuals = y_test - y_pred

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.show()


# ============================================================
# 23. RESIDUAL HISTOGRAM
# ============================================================

plt.hist(
    residuals,
    bins=10
)

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution")

plt.show()


# ============================================================
# 24. CATEGORICAL VS NUMERICAL - BAR
# ============================================================

sns.barplot(
    x="category",
    y="value",
    data=df
)

plt.show()


# ============================================================
# 25. CATEGORICAL VS NUMERICAL - BOX
# ============================================================

sns.boxplot(
    x="category",
    y="value",
    data=df
)

plt.show()


# ============================================================
# 26. NUMERICAL VS NUMERICAL - SCATTER
# ============================================================

sns.scatterplot(
    x="feature1",
    y="feature2",
    data=df
)

plt.show()


# ============================================================
# 27. NUMERICAL DISTRIBUTION - HISTOGRAM
# ============================================================

sns.histplot(
    df["feature1"],
    kde=True
)

plt.show()


# ============================================================
# 28. CORRELATION HEATMAP
# ============================================================

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.show()



f6


# ============================================================
# DATA SCIENCE LAB (CS2311) - ASSIGNMENT 2
# California Housing Dataset
# Q1: EDA + Preprocessing
# Q2: Multiple Linear Regression - 3 Models
# Q3: Feature Selection + Comparison
# ============================================================

# If needed:
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("housing.csv")

print("\n================ DATASET ================\n")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())


# ============================================================
# IDENTIFY TARGET
# ============================================================

# California Housing dataset target
target = "MedHouseVal"

X = df.drop(columns=[target])
y = df[target]

print("\nTarget variable:", target)
print("Predictor variables:")
print(X.columns.tolist())


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\n================ MISSING VALUES ================\n")

# Fill numerical missing values with median
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

print("Missing values after treatment:")
print(X.isnull().sum())


# ============================================================
# REMOVE DUPLICATES
# ============================================================

print("\n================ DUPLICATES ================\n")

data = pd.concat([X, y], axis=1)

print("Duplicates before:", data.duplicated().sum())

data = data.drop_duplicates()

print("Duplicates after:", data.duplicated().sum())

X = data.drop(columns=[target])
y = data[target]


# ============================================================
# OUTLIER DETECTION USING IQR
# ============================================================

print("\n================ OUTLIERS ================\n")

# Detect outliers in numerical predictor variables
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)

IQR = Q3 - Q1

outlier_count = ((X < (Q1 - 1.5 * IQR)) |
                 (X > (Q3 + 1.5 * IQR))).sum()

print("Outlier count in each feature:")
print(outlier_count)


# Treat outliers using clipping
# This keeps the records but limits extreme values
X_clean = X.copy()

for col in X_clean.columns:
    lower = Q1[col] - 1.5 * IQR[col]
    upper = Q3[col] + 1.5 * IQR[col]

    X_clean[col] = X_clean[col].clip(lower, upper)

print("\nOutliers treated using IQR clipping.")


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n================ CORRELATION ================\n")

corr = data.corr(numeric_only=True)

print("Correlation with target:")
print(corr[target].sort_values(ascending=False))


# ============================================================
# VISUALIZATION 1 - HISTOGRAMS
# ============================================================

X_clean.hist(figsize=(14, 10), bins=30)

plt.suptitle("Histograms of Housing Features")
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 2 - SCATTER PLOTS
# ============================================================

# Scatter plots of important features against target

top_features = (
    corr[target]
    .drop(target)
    .abs()
    .sort_values(ascending=False)
    .head(3)
    .index
)

for col in top_features:

    plt.figure(figsize=(7, 5))

    plt.scatter(X_clean[col], y, alpha=0.3)

    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(col + " vs " + target)

    plt.show()


# ============================================================
# VISUALIZATION 3 - BOX PLOTS
# ============================================================

plt.figure(figsize=(14, 7))

X_clean.boxplot()

plt.title("Box Plot of Housing Features")
plt.xticks(rotation=45)
plt.show()


# ============================================================
# VISUALIZATION 4 - CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X = X_clean

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n================ TRAIN TEST SPLIT ================\n")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# FEATURE SCALING
# ============================================================

# Standardization
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

print("\nFeature scaling completed using StandardScaler.")


# ============================================================
# Q2 - MODEL A
# 3 MOST HIGHLY CORRELATED FEATURES
# ============================================================

correlations = X.corrwith(y).abs().sort_values(ascending=False)

features_A = correlations.head(3).index.tolist()

print("\n================ MODEL A ================\n")

print("Features:")
print(features_A)


# ============================================================
# Q2 - MODEL B
# 5 SELECTED FEATURES
# ============================================================

features_B = correlations.head(5).index.tolist()

print("\n================ MODEL B ================\n")

print("Features:")
print(features_B)


# ============================================================
# Q2 - MODEL C
# ALL FEATURES
# ============================================================

features_C = X.columns.tolist()

print("\n================ MODEL C ================\n")

print("Features:")
print(features_C)


# ============================================================
# FUNCTION FOR MODEL TRAINING AND EVALUATION
# ============================================================

def train_model(features):

    model = LinearRegression()

    start_time = time.time()

    model.fit(
        X_train_scaled[features],
        y_train
    )

    end_time = time.time()

    y_pred = model.predict(
        X_test_scaled[features]
    )

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    execution_time = end_time - start_time

    return model, mae, mse, rmse, r2, execution_time


# ============================================================
# TRAIN MODEL A
# ============================================================

model_A, mae_A, mse_A, rmse_A, r2_A, time_A = train_model(features_A)


# ============================================================
# TRAIN MODEL B
# ============================================================

model_B, mae_B, mse_B, rmse_B, r2_B, time_B = train_model(features_B)


# ============================================================
# TRAIN MODEL C
# ============================================================

model_C, mae_C, mse_C, rmse_C, r2_C, time_C = train_model(features_C)


# ============================================================
# Q2 - COMPARATIVE TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Model A - Top 3",
        "Model B - Top 5",
        "Model C - All Features"
    ],

    "Number of Features": [
        len(features_A),
        len(features_B),
        len(features_C)
    ],

    "MAE": [
        mae_A,
        mae_B,
        mae_C
    ],

    "MSE": [
        mse_A,
        mse_B,
        mse_C
    ],

    "RMSE": [
        rmse_A,
        rmse_B,
        rmse_C
    ],

    "R2 Score": [
        r2_A,
        r2_B,
        r2_C
    ],

    "Training Time": [
        time_A,
        time_B,
        time_C
    ]
})

print("\n================ MODEL COMPARISON ================\n")

print(results)


# ============================================================
# BEST MODEL BASED ON R2
# ============================================================

best_model = results.loc[
    results["R2 Score"].idxmax()
]

print("\nBest model based on R2 Score:")
print(best_model)


# ============================================================
# Q3 - FEATURE SELECTION USING RFE
# ============================================================

print("\n================ RFE FEATURE SELECTION ================\n")

# Select 5 important features using RFE

rfe_model = LinearRegression()

rfe = RFE(
    estimator=rfe_model,
    n_features_to_select=5
)

rfe.fit(
    X_train_scaled,
    y_train
)

selected_features = X_train_scaled.columns[rfe.support_].tolist()

print("Selected features using RFE:")
print(selected_features)


# ============================================================
# RETRAIN MODEL USING RFE FEATURES
# ============================================================

rfe_final_model = LinearRegression()

start_time = time.time()

rfe_final_model.fit(
    X_train_scaled[selected_features],
    y_train
)

end_time = time.time()

rfe_predictions = rfe_final_model.predict(
    X_test_scaled[selected_features]
)


# ============================================================
# RFE MODEL METRICS
# ============================================================

rfe_mae = mean_absolute_error(
    y_test,
    rfe_predictions
)

rfe_mse = mean_squared_error(
    y_test,
    rfe_predictions
)

rfe_rmse = np.sqrt(rfe_mse)

rfe_r2 = r2_score(
    y_test,
    rfe_predictions
)

rfe_time = end_time - start_time


# ============================================================
# Q3 - SELECTED VS ALL FEATURES
# ============================================================

feature_selection_results = pd.DataFrame({

    "Model": [
        "RFE Selected Features",
        "All Features"
    ],

    "Number of Features": [
        len(selected_features),
        len(features_C)
    ],

    "MAE": [
        rfe_mae,
        mae_C
    ],

    "MSE": [
        rfe_mse,
        mse_C
    ],

    "RMSE": [
        rfe_rmse,
        rmse_C
    ],

    "R2 Score": [
        rfe_r2,
        r2_C
    ],

    "Training Time": [
        rfe_time,
        time_C
    ]
})

print("\n================ FEATURE SELECTION COMPARISON ================\n")

print(feature_selection_results)


# ============================================================
# RFE FEATURE IMPORTANCE
# ============================================================

print("\nRFE Ranking of Features:")

rfe_ranking = pd.DataFrame({

    "Feature": X.columns,

    "Rank": rfe.ranking_,

    "Selected": rfe.support_

})

print(rfe_ranking.sort_values("Rank"))


# ============================================================
# REGRESSION COEFFICIENTS
# ============================================================

print("\n================ RFE MODEL COEFFICIENTS ================\n")

coefficients = pd.DataFrame({

    "Feature": selected_features,

    "Coefficient": rfe_final_model.coef_

})

print(coefficients)


# ============================================================
# FINAL PREDICTION GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    rfe_predictions,
    alpha=0.4
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")

plt.title("Actual vs Predicted House Value - RFE Model")

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("FINAL SUMMARY")
print("============================================================")

print("\nModel A Features:")
print(features_A)

print("\nModel B Features:")
print(features_B)

print("\nModel C Features:")
print(features_C)

print("\nRFE Selected Features:")
print(selected_features)

print("\nModel Performance:")
print(results)

print("\nFeature Selection Performance:")
print(feature_selection_results)

print("\nUse the R2 Score, MAE, MSE and RMSE values above")
print("to discuss which feature combination performed better.")