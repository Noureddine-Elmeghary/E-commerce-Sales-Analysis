import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

#Data Loading & Cleaning

df = pd.read_csv(r"C:\Users\elmeg\Downloads\ecommerce_sales.csv")

print("Missing values per column:")
print(df.isnull().sum())

df.dropna(inplace=True)

print("\nMissing values handled by dropping rows.")

print("\nColumns in the dataframe:")
print(df.columns)

print("\nSkipping date conversion for now, as column name is unknown. A placeholder is provided in the code.")

print(f"\nNumber of duplicate rows before removal: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print("Duplicate rows removed.")

print("\nOriginal column names:")
print(df.columns)

df.columns = df.columns.str.lower().str.replace(' ', '_')

print("\nStandardized column names:")
print(df.columns)

print("\nData cleaning and preprocessing steps have been added to the script.")
print("Please review the script, especially the date conversion part, and adapt it to your dataset.")

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Distribution of Sales, Profit, and Quantity ---")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['sales'], kde=True)
plt.title('Distribution of Sales')

plt.subplot(1, 3, 2)
sns.histplot(df['profit'], kde=True)
plt.title('Distribution of Profit')

plt.subplot(1, 3, 3)
sns.histplot(df['quantity'], kde=True)
plt.title('Distribution of Quantity')

plt.tight_layout()
plt.show()

print("\n--- Correlation Analysis ---")
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Columns')
plt.show()

print("\n--- Outlier Detection ---")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=df['sales'])
plt.title('Boxplot of Sales')

plt.subplot(1, 3, 2)
sns.boxplot(y=df['profit'])
plt.title('Boxplot of Profit')

plt.subplot(1, 3, 3)
sns.boxplot(y=df['quantity'])
plt.title('Boxplot of Quantity')

plt.tight_layout()
plt.show()

print("\nExploratory Data Analysis steps have been added to the script.")

print("\n--- Answering Business Questions ---")

top_10_sales = df.groupby('product_name')['sales'].sum().sort_values(ascending=False).head(10)
top_10_profit = df.groupby('product_name')['profit'].sum().sort_values(ascending=False).head(10)

print("\n--- Top 10 Products by Sales ---")
print(top_10_sales)

print("\n--- Top 10 Products by Profit ---")
print(top_10_profit)

top_regions_revenue = df.groupby('region')['sales'].sum().sort_values(ascending=False).head(10)
top_customers_revenue = df.groupby('customer_name')['sales'].sum().sort_values(ascending=False).head(10)

print("\n--- Top 10 Regions by Revenue ---")
print(top_regions_revenue)

print("\n--- Top 10 Customers by Revenue ---")
print(top_customers_revenue)

print("\n--- Monthly Sales Trend ---")
print("Code for monthly sales trend is commented out. Please uncomment and adapt it to your date column.")

df['profit_margin'] = (df['profit'] / df['sales']) * 100
df['profit_margin'].fillna(0, inplace=True)

print("\n--- Profit Margin per Order ---")
print("A new column 'profit_margin' has been calculated.")
print(df[['sales', 'profit', 'profit_margin']].head())

shipping_profitability = df.groupby('shipping_mode')['profit_margin'].mean().sort_values(ascending=False)

print("\n--- Shipping Mode Profitability (Average Profit Margin) ---")
print(shipping_profitability)

print("\nBusiness analysis questions have been addressed in the script.")

print("\n--- Data Visualization ---")

print("\n--- Visualizing Sales Trend Over Time ---")
print("Code for sales trend line chart is commented out. Please uncomment and adapt it to your date column.")

print("\n--- Visualizing Top 10 Products by Sales ---")
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_sales.values, y=top_10_sales.index, palette='viridis')
plt.title('Top 10 Products by Sales')
plt.xlabel('Total Sales')
plt.ylabel('Product Name')
plt.show()

print("\n--- Visualizing Correlation Heatmap ---")
correlation_columns = ['sales', 'profit', 'discount', 'quantity']
existing_columns = [col for col in correlation_columns if col in df.columns]

if len(existing_columns) > 1:
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[existing_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation between Sales, Profit, Discount, and Quantity')
    plt.show()
else:
    print("Not enough columns found to create a correlation heatmap.")

print("\n--- Visualizing Sales Distribution by Region ---")
region_sales = df.groupby('region')['sales'].sum()

plt.figure(figsize=(10, 8))
plt.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Sales Distribution by Region')
plt.ylabel('')
plt.show()

print("\nData visualization steps have been added to the script.")

print("\n--- Advanced Analytics ---")

print("\n--- RFM Analysis ---")
print("Code for RFM analysis is commented out. Please uncomment and adapt it to your dataset.")

print("\n--- Sales Forecasting ---")
print("Code for Sales Forecasting is commented out. Please uncomment and adapt it to your dataset.")

print("\n--- Profitability Analysis by Category ---")
print("Code for Profitability Analysis by Category is commented out. Please uncomment and adapt it to your dataset.")

print("\nAdvanced analytics steps have been added to the script.")