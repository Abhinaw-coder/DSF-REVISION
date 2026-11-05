#5
!pip install numpy pandas scipy matplotlib seaborn

# QUESTION 1: Probability Distributions - Binomial and Poisson

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson

# --------------------------------------------------
# PART A: BINOMIAL DISTRIBUTION
# --------------------------------------------------

n = 20
p = 0.70

# 1. Define Binomial distribution
print("PART A: BINOMIAL DISTRIBUTION")
print("Distribution: X ~ Binomial(n=20, p=0.70)")

# 2. Calculate probabilities
exactly_15 = binom.pmf(15, n, p)
at_least_16 = binom.sf(15, n, p)
fewer_than_10 = binom.cdf(9, n, p)

print("\nProbability Calculations:")
print(f"Exactly 15 patients respond: {exactly_15:.6f}")
print(f"At least 16 patients respond: {at_least_16:.6f}")
print(f"Fewer than 10 patients respond: {fewer_than_10:.6f}")

# 3. Expected value and variance
expected_value = binom.mean(n, p)
variance = binom.var(n, p)

print("\nExpected Number of Positive Responses:", expected_value)
print("Variance:", variance)

# 4. Generate probability distribution
x = np.arange(0, n + 1)
probabilities = binom.pmf(x, n, p)

binomial_table = pd.DataFrame({
    "Number of Positive Responses": x,
    "Probability": probabilities
})

print("\nBinomial Probability Distribution:")
print(binomial_table.to_string(index=False))

# Plot Binomial Distribution
plt.figure(figsize=(10, 5))
plt.bar(x, probabilities)
plt.xlabel("Number of Positive Responses")
plt.ylabel("Probability")
plt.title("Binomial Probability Distribution")
plt.xticks(x)
plt.grid(axis="y", alpha=0.3)
plt.show()

# 5. Explanation
print("\nWhy Binomial Distribution is Appropriate:")
print("""
1. There is a fixed number of trials (20 patients).
2. Each patient has two possible outcomes:
   Positive response or Negative response.
3. The probability of success is constant (0.70).
4. The outcomes are assumed to be independent.

Therefore, the Binomial distribution is appropriate.
""")


# --------------------------------------------------
# PART B: POISSON DISTRIBUTION
# --------------------------------------------------

print("\n\nPART B: POISSON DISTRIBUTION")

# Number of arrivals during 20 consecutive intervals
arrivals = np.array([
    4, 7, 5, 6, 3, 8, 5, 4, 6, 7,
    5, 3, 4, 6, 5, 8, 4, 5, 6, 7
])

# 1. Calculate average arrival rate
lam = np.mean(arrivals)

print("\nAverage Arrival Rate (Lambda):", lam)
print("Distribution: X ~ Poisson(lambda =", lam, ")")

# 2 and 3. Calculate probabilities
exactly_5 = poisson.pmf(5, lam)
fewer_than_4 = poisson.cdf(3, lam)
more_than_7 = poisson.sf(7, lam)

print("\nProbability Calculations:")
print(f"Exactly 5 arrivals: {exactly_5:.6f}")
print(f"Fewer than 4 arrivals: {fewer_than_4:.6f}")
print(f"More than 7 arrivals: {more_than_7:.6f}")

# 4. Compare observed and theoretical frequencies
unique_values, observed_counts = np.unique(
    arrivals, return_counts=True
)

# Include possible values from 0 to 12
x_poisson = np.arange(0, 13)

observed_frequency = np.array([
    np.sum(arrivals == value) for value in x_poisson
])

theoretical_probability = poisson.pmf(x_poisson, lam)

# Expected frequency = Probability * Total observations
expected_frequency = theoretical_probability * len(arrivals)

comparison = pd.DataFrame({
    "Number of Arrivals": x_poisson,
    "Observed Frequency": observed_frequency,
    "Theoretical Probability": theoretical_probability,
    "Expected Frequency": expected_frequency
})

print("\nObserved vs Theoretical Frequencies:")
print(comparison.round(4).to_string(index=False))

# Plot observed and theoretical frequencies
plt.figure(figsize=(10, 5))

plt.bar(
    x_poisson - 0.2,
    observed_frequency,
    width=0.4,
    label="Observed Frequency"
)

plt.bar(
    x_poisson + 0.2,
    expected_frequency,
    width=0.4,
    label="Theoretical Frequency"
)

plt.xlabel("Number of Patient Arrivals")
plt.ylabel("Frequency")
plt.title("Observed vs Theoretical Poisson Frequencies")
plt.xticks(x_poisson)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.show()

# 5. Model suitability
print("\nComment on Poisson Distribution:")
print("""
The Poisson distribution is appropriate for modeling the
number of patient arrivals during a fixed time interval.

It assumes:
1. Events occur independently.
2. The average arrival rate remains constant.
3. Events occur individually within the interval.

The observed frequencies can be compared with the
theoretical frequencies to evaluate how well the model fits.

The Poisson distribution provides a reasonable starting
model, although the small sample size of 20 intervals
limits the strength of the conclusion.
""")


# QUESTION 2: Normal Distribution and Basic Statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statistics import multimode

# --------------------------------------------------
# DATA: EXAMINATION MARKS OF 30 STUDENTS
# --------------------------------------------------

marks = np.array([
    56, 72, 68, 81, 75, 63, 91, 48, 67, 73,
    85, 59, 77, 69, 54, 88, 71, 64, 79, 82,
    61, 74, 93, 57, 66, 76, 70, 52, 84, 62
])

print("QUESTION 2: NORMAL DISTRIBUTION AND BASIC STATISTICS")

# --------------------------------------------------
# 1. DESCRIPTIVE STATISTICS
# --------------------------------------------------

mean = np.mean(marks)
median = np.median(marks)

modes = multimode(marks)

minimum = np.min(marks)
maximum = np.max(marks)
data_range = maximum - minimum

# Population variance and standard deviation
variance = np.var(marks, ddof=0)
std_dev = np.std(marks, ddof=0)

# Quartiles
Q1 = np.percentile(marks, 25)
Q3 = np.percentile(marks, 75)

IQR = Q3 - Q1

print("\n1. DESCRIPTIVE STATISTICS")
print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Mode: {modes if len(modes) < len(marks) else 'No mode'}")
print(f"Minimum: {minimum}")
print(f"Maximum: {maximum}")
print(f"Range: {data_range}")
print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"First Quartile (Q1): {Q1:.2f}")
print(f"Third Quartile (Q3): {Q3:.2f}")
print(f"Interquartile Range (IQR): {IQR:.2f}")

# --------------------------------------------------
# 2. Z-SCORE OF STUDENT SCORING 91
# --------------------------------------------------

student_mark = 91

z_score = (student_mark - mean) / std_dev

print("\n2. Z-SCORE")
print(f"Student Mark: {student_mark}")
print(f"Z-score: {z_score:.4f}")

# --------------------------------------------------
# 3. NORMAL DISTRIBUTION PROBABILITIES
# --------------------------------------------------

# Probability of scoring more than 80
prob_more_than_80 = norm.sf(80, loc=mean, scale=std_dev)

# Probability of scoring less than 60
prob_less_than_60 = norm.cdf(60, loc=mean, scale=std_dev)

# Probability of scoring between 60 and 80
prob_between_60_80 = (
    norm.cdf(80, loc=mean, scale=std_dev)
    - norm.cdf(60, loc=mean, scale=std_dev)
)

print("\n3. NORMAL DISTRIBUTION PROBABILITIES")
print(f"P(X > 80): {prob_more_than_80:.4f}")
print(f"P(X < 60): {prob_less_than_60:.4f}")
print(f"P(60 < X < 80): {prob_between_60_80:.4f}")

# --------------------------------------------------
# 4. IDENTIFY UNUSUAL STUDENTS USING |Z| > 2
# --------------------------------------------------

z_scores = (marks - mean) / std_dev

unusual_students = marks[np.abs(z_scores) > 2]

print("\n4. UNUSUAL OBSERVATIONS")
print("Marks with |Z| > 2:", unusual_students)

if len(unusual_students) == 0:
    print("No students have unusual marks according to the given threshold.")
else:
    print("Unusual marks identified.")

# Display all marks and their z-scores
z_table = pd.DataFrame({
    "Student": [f"Student {i+1}" for i in range(len(marks))],
    "Marks": marks,
    "Z-Score": z_scores
})

print("\nZ-Score Table:")
print(z_table.round(4).to_string(index=False))

# --------------------------------------------------
# 5. ACTUAL VS PREDICTED PROPORTION ABOVE 80
# --------------------------------------------------

actual_count = np.sum(marks > 80)
actual_proportion = actual_count / len(marks)

predicted_proportion = prob_more_than_80

print("\n5. ACTUAL VS PREDICTED PROPORTION")
print(f"Number of students scoring above 80: {actual_count}")
print(f"Actual proportion: {actual_proportion:.4f}")
print(f"Predicted Normal proportion: {predicted_proportion:.4f}")

print(f"Actual percentage: {actual_proportion * 100:.2f}%")
print(f"Predicted percentage: {predicted_proportion * 100:.2f}%")

# --------------------------------------------------
# 6. VISUALIZATIONS
# --------------------------------------------------

# Histogram with Normal Distribution Curve
plt.figure(figsize=(10, 5))

plt.hist(
    marks,
    bins=8,
    density=True,
    alpha=0.6,
    edgecolor="black",
    label="Observed Marks"
)

x = np.linspace(minimum - 5, maximum + 5, 200)
normal_curve = norm.pdf(x, loc=mean, scale=std_dev)

plt.plot(
    x,
    normal_curve,
    linewidth=2,
    label="Normal Distribution Curve"
)

plt.xlabel("Examination Marks")
plt.ylabel("Density")
plt.title("Distribution of Examination Marks")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Box Plot
plt.figure(figsize=(8, 4))

plt.boxplot(marks, vert=False)

plt.xlabel("Examination Marks")
plt.title("Box Plot of Student Examination Marks")
plt.grid(axis="x", alpha=0.3)
plt.show()

# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------

print("\n6. OVERALL INTERPRETATION")
print(f"""
1. The mean examination score is {mean:.2f}.
2. The median examination score is {median:.2f}.
3. The standard deviation is {std_dev:.2f}, indicating
   the spread of marks around the mean.
4. The range of marks is {data_range}.
5. The z-score of 91 is {z_score:.4f}.
6. Students with an absolute z-score greater than 2
   are considered unusual.
7. The actual proportion scoring above 80 is
   {actual_proportion * 100:.2f}%.
8. The Normal distribution predicts approximately
   {predicted_proportion * 100:.2f}% scoring above 80.

The histogram and box plot help examine the distribution
and spread of examination marks. The Normal distribution
provides an approximate model of student performance.
""")

# QUESTION 3: Correlation and Covariance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# DATA: 12 STUDENTS
# --------------------------------------------------

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 4, 7, 9],

    "Attendance": [62, 68, 71, 75, 78, 82, 85, 88, 91, 65, 79, 86],

    "Final_Marks": [48, 52, 58, 61, 67, 72, 76, 81, 85, 55, 70, 79]
}

df = pd.DataFrame(data)

print("QUESTION 3: CORRELATION AND COVARIANCE")

print("\nOriginal Dataset:")
print(df.to_string(index=False))

# --------------------------------------------------
# 1. MEAN AND STANDARD DEVIATION
# --------------------------------------------------

means = df.mean()

# Sample standard deviation
std_deviations = df.std(ddof=1)

statistics_table = pd.DataFrame({
    "Mean": means,
    "Standard Deviation": std_deviations
})

print("\n1. MEAN AND STANDARD DEVIATION")
print(statistics_table.round(4))

# --------------------------------------------------
# 2. COVARIANCE MATRIX
# --------------------------------------------------

covariance_matrix = df.cov()

print("\n2. COVARIANCE MATRIX")
print(covariance_matrix.round(4))

# --------------------------------------------------
# 3. PEARSON CORRELATION COEFFICIENTS
# --------------------------------------------------

correlation_matrix = df.corr(method="pearson")

print("\n3. PEARSON CORRELATION MATRIX")
print(correlation_matrix.round(4))

# Individual correlation coefficients
study_marks_corr = df["Study_Hours"].corr(df["Final_Marks"])

attendance_marks_corr = df["Attendance"].corr(df["Final_Marks"])

study_attendance_corr = df["Study_Hours"].corr(df["Attendance"])

print("\nIndividual Correlation Coefficients:")

print(
    f"Study Hours and Final Marks: "
    f"{study_marks_corr:.4f}"
)

print(
    f"Attendance and Final Marks: "
    f"{attendance_marks_corr:.4f}"
)

print(
    f"Study Hours and Attendance: "
    f"{study_attendance_corr:.4f}"
)

# --------------------------------------------------
# 4. IDENTIFY STRONGEST LINEAR RELATIONSHIP
# --------------------------------------------------

correlations = {
    "Study Hours and Final Marks": study_marks_corr,
    "Attendance and Final Marks": attendance_marks_corr,
    "Study Hours and Attendance": study_attendance_corr
}

strongest_pair = max(
    correlations,
    key=lambda pair: abs(correlations[pair])
)

print("\n4. STRONGEST LINEAR RELATIONSHIP")

print("Strongest Pair:", strongest_pair)

print(
    "Correlation Coefficient:",
    round(correlations[strongest_pair], 4)
)

# --------------------------------------------------
# 5. VISUALIZATIONS
# --------------------------------------------------

def create_scatter_plot(x, y, xlabel, ylabel, title):

    plt.figure(figsize=(8, 5))

    plt.scatter(x, y, s=70, label="Student Data")

    # Calculate regression line
    slope, intercept = np.polyfit(x, y, 1)

    regression_line = slope * np.array(x) + intercept

    plt.plot(
        x,
        regression_line,
        linewidth=2,
        label="Regression Line"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# Study Hours vs Final Marks
create_scatter_plot(
    df["Study_Hours"],
    df["Final_Marks"],
    "Study Hours per Week",
    "Final Marks",
    "Study Hours vs Final Marks"
)

# Attendance vs Final Marks
create_scatter_plot(
    df["Attendance"],
    df["Final_Marks"],
    "Attendance (%)",
    "Final Marks",
    "Attendance vs Final Marks"
)

# Study Hours vs Attendance
create_scatter_plot(
    df["Study_Hours"],
    df["Attendance"],
    "Study Hours per Week",
    "Attendance (%)",
    "Study Hours vs Attendance"
)

# --------------------------------------------------
# ADDITIONAL VISUALIZATION: CORRELATION HEATMAP
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation Coefficient")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=20
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):

        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. INTERPRETATION OF CORRELATION COEFFICIENTS
# --------------------------------------------------

def interpret_correlation(r):

    if r >= 0.7:
        return "Strong Positive Correlation"

    elif r >= 0.3:
        return "Moderate Positive Correlation"

    elif r > 0:
        return "Weak Positive Correlation"

    elif r <= -0.7:
        return "Strong Negative Correlation"

    elif r <= -0.3:
        return "Moderate Negative Correlation"

    elif r < 0:
        return "Weak Negative Correlation"

    else:
        return "No Linear Correlation"


print("\n6. INTERPRETATION OF CORRELATIONS")

for pair, value in correlations.items():

    print(f"\n{pair}: {value:.4f}")

    print("Interpretation:", interpret_correlation(value))

# --------------------------------------------------
# 7. COMPARISON OF COVARIANCE AND CORRELATION
# --------------------------------------------------

print("\n7. COVARIANCE VS CORRELATION")

print("\nCovariance Matrix:")
print(covariance_matrix.round(4))

print("\nCorrelation Matrix:")
print(correlation_matrix.round(4))

print("""
Covariance indicates the direction in which two variables
change together.

Positive covariance indicates that two variables tend
to increase together.

Negative covariance indicates that one variable tends
to increase when the other decreases.

Correlation measures both the direction and strength
of a linear relationship.

Correlation values always lie between -1 and +1,
whereas covariance values are not restricted to this range.

Covariance depends on the units and scales of the variables.
Therefore, covariance values cannot be directly compared
across variables having different scales.

Correlation is standardized and unit-free, making it
more suitable for comparing relationships.
""")

# --------------------------------------------------
# 8. FINAL CONCLUSION: CORRELATION VS CAUSATION
# --------------------------------------------------

print("\n8. FINAL CONCLUSION")

print(f"""
Study Hours and Final Marks:
Correlation = {study_marks_corr:.4f}

Attendance and Final Marks:
Correlation = {attendance_marks_corr:.4f}

Study Hours and Attendance:
Correlation = {study_attendance_corr:.4f}

CONCLUSION:

The analysis indicates positive relationships between
study hours, attendance, and final examination marks.

Students who study for more hours per week generally
tend to obtain higher examination marks.

Similarly, students with higher attendance percentages
generally tend to obtain higher examination marks.

Study hours and attendance also show a positive
relationship with each other.

However, correlation does not necessarily imply causation.

These results show associations between the variables
but do not prove that increased study hours or attendance
directly cause higher examination marks.

Other factors, such as learning ability, teaching quality,
study methods, and prior knowledge, may also influence
academic performance.
""")


#6
# QUESTION 1: STUDENT PERFORMANCE ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load UCI Student Performance Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
df = pd.read_csv(url, sep=";")

# a. Dataset information
print("Dimensions:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())

# b. Data preprocessing
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nAfter Preprocessing:")
print("Missing Values:\n", df.isnull().sum())

# Identify outliers using IQR
cols = ["studytime", "absences", "G1", "G2", "G3"]

for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) |
                  (df[col] > Q3 + 1.5 * IQR)]
    print(col, "Outliers:", len(outliers))

# c. Statistical analysis
print("\nStatistical Analysis:")
for col in ["studytime", "absences", "G1", "G2", "G3"]:
    print("\n", col)
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Standard Deviation:", df[col].std())
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())
    print("IQR:", df[col].quantile(0.75) - df[col].quantile(0.25))

# d. Pearson correlation
pairs = [
    ("studytime", "G3"),
    ("G1", "G3"),
    ("G2", "G3"),
    ("absences", "G3")
]

print("\nPearson Correlation:")
for x, y in pairs:
    print(x, "and", y, ":", df[x].corr(df[y]))

# e. Visualizations

# 1. Correlation Heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(df.select_dtypes(include="number").corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 2. Study Time vs Final Grade
sns.scatterplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.show()

# 3. Previous Grade vs Final Grade
sns.scatterplot(data=df, x="G2", y="G3")
plt.title("G2 vs Final Grade")
plt.show()

# 4. Box Plot of Final Grades
sns.boxplot(x=df["G3"])
plt.title("Final Grade Distribution")
plt.show()

# f. Strongest relationship with final performance
correlations = df.select_dtypes(include="number").corr()["G3"].drop("G3")
strongest = correlations.abs().idxmax()

print("\nStrongest Relationship with Final Grade:")
print(strongest, ":", correlations[strongest])

print("""
Conclusion:
Previous examination grades show a strong relationship
with final performance. Study time and attendance-related
factors can also influence student performance.
""")

# QUESTION 2: BANK MARKETING CLASSIFICATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Load Dataset (download bank-full.csv from UCI)
df = pd.read_csv("bank-full.csv", sep=";")

# a. Dataset examination
print("Dimensions:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe(include="all"))
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())
print("\nTarget Distribution:\n", df["y"].value_counts())

# b. Data preprocessing
df = df.drop_duplicates().copy()

X = df.drop("y", axis=1)
y = df["y"].map({"yes": 1, "no": 0})

numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

print("\nPreprocessing: Missing values imputed, categorical variables encoded and numerical variables scaled.")

# c. Exploratory Data Analysis

# 1. Target Distribution
sns.countplot(data=df, x="y")
plt.title("Term Deposit Subscription Distribution")
plt.show()

# 2. Age vs Subscription
sns.boxplot(data=df, x="y", y="age")
plt.title("Age vs Subscription")
plt.show()

# 3. Job vs Subscription
sns.countplot(data=df, x="job", hue="y")
plt.title("Job vs Subscription")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# d. Train-test split and classification models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

# e. Train and evaluate models
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append([
        name, accuracy, precision, recall, f1
    ])

    print("\n", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=["No", "Yes"],
        cmap="Blues"
    )
    plt.title(name + " Confusion Matrix")
    plt.show()

# f. Comparative performance table
comparison = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nModel Comparison:")
print(comparison.to_string(index=False))

best_model = comparison.loc[comparison["F1 Score"].idxmax()]

print("\nBest Model:")
print(best_model)

# g. Factors affecting subscription
print("\nSubscription Rate by Job:")
print(df.groupby("job")["y"].apply(
    lambda x: (x == "yes").mean()
).sort_values(ascending=False))

print("""
Conclusion:
The three classification models are evaluated using
accuracy, precision, recall and F1-score.

The model with the highest F1-score provides the best
balance between precision and recall in this experiment.

Customer attributes such as age, job, education, contact
method and previous campaign outcomes may influence
subscription decisions.

The best model is selected based on the experimental
results rather than assuming one model is always superior.
""")

# QUESTION 3: PAIRED T-TEST AND TRAINING EFFECTIVENESS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, norm

# Given Data
before = np.array([54, 61, 58, 67, 63, 55, 72, 69, 60, 64, 57, 71])
after = np.array([59, 65, 63, 72, 68, 61, 76, 74, 65, 69, 62, 77])

# a. Mean and Standard Deviation
print("Before Training")
print("Mean:", np.mean(before))
print("Standard Deviation:", np.std(before, ddof=1))

print("\nAfter Training")
print("Mean:", np.mean(after))
print("Standard Deviation:", np.std(after, ddof=1))

# b. Paired t-test
t_stat, p_value = ttest_rel(after, before)

print("\nPaired t-test")
print("T-statistic:", t_stat)
print("P-value:", p_value)

# c. Statistical significance
alpha = 0.05

if p_value < alpha:
    print("There is a statistically significant improvement.")
else:
    print("There is no statistically significant improvement.")

# d. Difference and Normal Distribution
difference = after - before

mean_diff = np.mean(difference)
std_diff = np.std(difference, ddof=1)

probability = norm.sf(5, loc=mean_diff, scale=std_diff)

print("\nScore Differences:")
print(difference)

print("Mean Difference:", mean_diff)
print("Standard Deviation of Differences:", std_diff)
print("Probability of Improvement > 5:", probability)

# e. Visualization

# 1. Before vs After Box Plot
df = pd.DataFrame({
    "Before": before,
    "After": after
})

sns.boxplot(data=df)
plt.title("Before and After Training Scores")
plt.ylabel("Programming Scores")
plt.show()

# 2. Individual Score Comparison
for i in range(len(before)):
    plt.plot([1, 2], [before[i], after[i]], marker="o")

plt.xticks([1, 2], ["Before Training", "After Training"])
plt.ylabel("Programming Scores")
plt.title("Employee Performance Before and After Training")
plt.grid(alpha=0.3)
plt.show()

# f. Final Conclusion
print("""
Conclusion:
The mean score after training is higher than the mean
score before training.

The paired t-test determines whether this improvement
is statistically significant at the 0.05 significance level.

The mean difference represents the average improvement,
while the probability calculation estimates the chance
of improvement exceeding 5 marks.

Statistical significance indicates that the observed
improvement is unlikely to be due to random variation
under the null hypothesis.

Practical improvement indicates whether the increase
in marks is meaningful in real-world performance.

Overall, the training program improved employee
programming performance.
""")


#7

pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn lifelines

# QUESTION 1: POISSON REGRESSION

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load and inspect dataset
df = sns.load_dataset("tips")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe(include="all"))

# Handle missing values
df = df.dropna()

# Split data into 80% training and 20% testing
train, test = train_test_split(
    df, test_size=0.2, random_state=42
)

# Build Poisson Regression Model
model = smf.glm(
    formula="size ~ total_bill + tip + C(sex) + C(smoker) + C(day) + C(time)",
    data=train,
    family=sm.families.Poisson()
).fit()

# Display model summary
print("\nPoisson Regression Summary:")
print(model.summary())

# Predict test observations
predictions = model.predict(test)

# Calculate MAE
mae = mean_absolute_error(test["size"], predictions)

print("\nMean Absolute Error:", mae)

# Compare actual and predicted values
result = pd.DataFrame({
    "Actual": test["size"],
    "Predicted": predictions
})

print("\nActual vs Predicted:\n", result.head(15))

# Plot actual vs predicted
plt.figure(figsize=(7, 5))
plt.scatter(test["size"], predictions)
plt.xlabel("Actual Party Size")
plt.ylabel("Predicted Party Size")
plt.title("Actual vs Predicted Party Size")
plt.grid()
plt.show()

# Interpretation
print("""
Conclusion:
Poisson regression is suitable because party size is a
count-type variable containing non-negative integers.

The model estimates the expected number of people using
the explanatory variables.

Regression coefficients show the direction of the
relationship, and p-values indicate statistical significance.
A p-value below 0.05 generally indicates significance.
MAE measures the average prediction error.
""")


# QUESTION 2: SURVIVAL ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_lung
from lifelines.statistics import logrank_test

# Load dataset
df = load_lung()

print("First 5 Rows:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Handle missing values
df = df.dropna(subset=["time", "status", "sex"])

# Identify survival duration and event indicator
# In the original lung dataset: 1 = censored, 2 = death
if set(df["status"].unique()).issubset({1, 2}):
    df["event"] = df["status"].map({1: 0, 2: 1})
else:
    df["event"] = df["status"]

# Overall Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(df["time"], event_observed=df["event"])

# Plot overall survival curve
kmf.plot_survival_function()
plt.title("Overall Kaplan-Meier Survival Curve")
plt.xlabel("Time (Days)")
plt.ylabel("Survival Probability")
plt.grid()
plt.show()

# Survival probabilities
print("\nSurvival Probabilities:")
for day in [100, 200, 365]:
    print(day, "days:", kmf.predict(day))

# Median survival time
print("\nOverall Median Survival Time:", kmf.median_survival_time_)

# Divide patients based on sex
male = df[df["sex"] == 1]
female = df[df["sex"] == 2]

# Fit separate Kaplan-Meier models
km_male = KaplanMeierFitter()
km_female = KaplanMeierFitter()

km_male.fit(male["time"], male["event"], label="Male")
km_female.fit(female["time"], female["event"], label="Female")

# Plot separate survival curves
plt.figure(figsize=(8, 5))
km_male.plot_survival_function()
km_female.plot_survival_function()
plt.title("Survival Curves by Sex")
plt.xlabel("Time (Days)")
plt.ylabel("Survival Probability")
plt.grid()
plt.show()

# Median survival times
print("\nMale Median Survival:", km_male.median_survival_time_)
print("Female Median Survival:", km_female.median_survival_time_)

# Log-rank test
result = logrank_test(
    male["time"],
    female["time"],
    event_observed_A=male["event"],
    event_observed_B=female["event"]
)

print("\nLog-Rank Test")
print("Test Statistic:", result.test_statistic)
print("P-value:", result.p_value)

if result.p_value < 0.05:
    print("Survival distributions differ significantly.")
else:
    print("No statistically significant difference in survival distributions.")

print("""
Conclusion:
The Kaplan-Meier estimator is used to estimate survival
probabilities over time while accounting for censored data.

The median survival time represents the time at which
the estimated survival probability reaches 50%.

The log-rank test compares survival distributions
between male and female patients.

A p-value below 0.05 indicates a statistically significant
difference between the two survival curves.
""")


# QUESTION 3: LINEAR AND POLYNOMIAL REGRESSION

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = sns.load_dataset("mpg")

print("First 5 Rows:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Remove missing horsepower values
df = df.dropna(subset=["horsepower"])

X = df[["horsepower"]]
y = df["mpg"]

# Scatter plot
plt.scatter(X, y)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Horsepower vs Fuel Efficiency")
plt.grid()
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []

# Simple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

results.append(["Linear Regression", rmse, r2])

# Plot linear regression
x_plot = np.linspace(X.min().iloc[0], X.max().iloc[0], 200).reshape(-1, 1)
plt.scatter(X, y, label="Actual Data")
plt.plot(x_plot, model.predict(x_plot), color="red", label="Linear")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Linear Regression")
plt.legend()
plt.show()

print("Linear Regression RMSE:", rmse)
print("Linear Regression R2:", r2)

# Polynomial Regression: Degrees 2, 3 and 4
for degree in [2, 3, 4]:

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    pred = model.predict(X_test_poly)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append([f"Polynomial Degree {degree}", rmse, r2])

    # Plot fitted polynomial curve
    x_poly = poly.transform(x_plot)
    y_poly = model.predict(x_poly)

    plt.scatter(X, y, label="Actual Data")
    plt.plot(x_plot, y_poly, color="red", label=f"Degree {degree}")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.title(f"Polynomial Regression Degree {degree}")
    plt.legend()
    plt.show()

    print(f"\nDegree {degree} RMSE:", rmse)
    print(f"Degree {degree} R2:", r2)

# Compare all models
comparison = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "R2 Score"]
)

print("\nModel Comparison:")
print(comparison.to_string(index=False))

# Best model
best = comparison.loc[comparison["RMSE"].idxmin()]

print("\nBest Model:")
print(best)

print("""
Conclusion:
Linear regression models a straight-line relationship
between horsepower and fuel efficiency.

Polynomial regression captures nonlinear relationships.

A lower RMSE and higher R2 indicate better performance.

Increasing polynomial degree does not always improve
the model. Higher degrees can cause overfitting,
where the model performs well on training data but
poorly on unseen data.
""")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target

X = df[["bmi"]]
y = df["target"]

plt.scatter(X, y)
plt.xlabel("BMI")
plt.ylabel("Diabetes Progression")
plt.title("BMI vs Diabetes Progression")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {}

linear = LinearRegression()
linear.fit(X_train, y_train)
models["Linear Regression"] = linear

for k in [3, 5, 7]:
    spline = SplineTransformer(n_knots=k, degree=3)
    X_train_spline = spline.fit_transform(X_train)
    X_test_spline = spline.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_spline, y_train)

    models[f"Spline {k} Knots"] = (spline, model)

results = []

for name, model in models.items():
    if name == "Linear Regression":
        predictions = model.predict(X_test)
    else:
        spline, reg = model
        predictions = reg.predict(spline.transform(X_test))

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append([name, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2 Score"])
print(results_df)

X_plot = pd.DataFrame({
    "bmi": np.linspace(X["bmi"].min(), X["bmi"].max(), 300)
})

plt.scatter(X, y, alpha=0.4, label="Observed Data")

for name, model in models.items():
    if name == "Linear Regression":
        y_plot = model.predict(X_plot)
    else:
        spline, reg = model
        y_plot = reg.predict(spline.transform(X_plot))

    plt.plot(X_plot["bmi"], y_plot, label=name)

plt.xlabel("BMI")
plt.ylabel("Diabetes Progression")
plt.title("Linear and Spline Regression")
plt.legend()
plt.show()

best_model = results_df.loc[results_df["RMSE"].idxmin()]
print("Best Model:", best_model["Model"])
print("Best RMSE:", best_model["RMSE"])
print("Best R2 Score:", best_model["R2 Score"])

print("\nConclusion:")
print("Spline regression can capture nonlinear relationships better than linear regression.")
print("Increasing the number of knots makes the model more flexible.")
print("However, too many knots may cause overfitting.")
print("The model with the lowest RMSE and highest R2 score performs best.")