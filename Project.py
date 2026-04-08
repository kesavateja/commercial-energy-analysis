import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------- LOAD DATA --------------------
df = pd.read_csv("Statewide_Commercial_Baseline_Study_of_New_York_Characteristics_of_Energy_Using_Equipment__2019.csv")

print(df.head())
print(df.info())

# -------------------- SELECT NUMERIC DATA --------------------
numeric_df = df.select_dtypes(include=np.number)

# Fill missing values (VERY IMPORTANT)
numeric_df = numeric_df.fillna(numeric_df.mean())

print("\nNumeric Columns:\n", numeric_df.columns)

# -------------------- HISTOGRAM --------------------
numeric_df.hist(figsize=(10,8))
plt.suptitle("Histogram of Numeric Features")
plt.show()

# -------------------- BAR CHART --------------------
df['Equipment Category'].value_counts().head(10).plot(kind='bar')
plt.title("Top Equipment Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# -------------------- LINE CHART --------------------
numeric_df['Weighted % with Response'].head(50).plot(kind='line')
plt.title("Weighted % with Response (Line Chart)")
plt.show()

# -------------------- PIE CHART --------------------
df['Region'].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%')
plt.title("Region Distribution")
plt.ylabel("")
plt.show()

# -------------------- BOX PLOT --------------------
sns.boxplot(data=numeric_df)
plt.title("Box Plot")
plt.show()

# -------------------- SCATTER PLOT --------------------
sns.scatterplot(
    x=numeric_df['Total Widgets in Enduse'],
    y=numeric_df['% of Equipment (Weighted)']
)
plt.title("Scatter Plot")
plt.show()

# -------------------- HEATMAP --------------------
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------- PAIRPLOT --------------------
sns.pairplot(numeric_df)
plt.show()

# -------------------- BUBBLE PLOT --------------------
plt.scatter(
    numeric_df['Total Widgets in Enduse'],
    numeric_df['% of Equipment (Weighted)'],
    s=numeric_df['Total N (Respondents)'] * 2,
    alpha=0.5
)
plt.title("Bubble Plot")
plt.xlabel("Total Widgets")
plt.ylabel("Equipment %")
plt.show()

# -------------------- STATISTICS --------------------
print("Mean:\n", numeric_df.mean())
print("Median:\n", numeric_df.median())
print("Variance:\n", numeric_df.var())
print("Correlation:\n", numeric_df.corr())

# -------------------- LINEAR REGRESSION --------------------
X = numeric_df.drop(columns=['% of Equipment (Weighted)'])
y = numeric_df['% of Equipment (Weighted)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))

# -------------------- CLASSIFICATION MODEL --------------------
# Convert target to categories
y_class = pd.cut(y, bins=3, labels=[0,1,2])

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
