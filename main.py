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

