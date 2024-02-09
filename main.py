import keras
import sklearn.model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf
from xgboost import XGBClassifier
from tensorflow import keras
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

tf.disable_v2_behavior()
import numpy as np
import pandas as pd


# TODO: may we drop the salary colmuon?

def bar_plot():
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='work_year', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by Work Year')
    plt.xlabel('work_year')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='job_title', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by Job Title')
    plt.xlabel('job_title')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='job_category', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by Job Category')
    plt.xlabel('Job Category')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='salary_currency', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by salary_currency')
    plt.xlabel('salary_currency')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='employee_residence', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by employee_residence')
    plt.xlabel('employee_residence')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='experience_level', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by experience_level')
    plt.xlabel('employee_residence')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='employment_type', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by employment_type')
    plt.xlabel('employment_type')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='work_setting', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by work_setting')
    plt.xlabel('work_setting')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='company_location', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by Company Location')
    plt.xlabel('Company Location')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='company_size', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by Company Size')
    plt.xlabel('Company Size')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()


def load_data_to_train_and_test():
    # Define features (X) and target variables (y)
    x = df.drop(columns=["salary", "salary_in_usd"])  # Features
    y = df[["salary", "salary_in_usd"]]  # Target variables

    # Split the data into training and testing sets
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Print the shapes of the training and testing sets to verify the split
    # print("Training set shape (X):", x_train.shape)
    # print("Testing set shape (X):", x_test.shape)
    # print("Training set shape (y):", y_train.shape)
    # print("Testing set shape (y):", y_test.shape)

    train_set.to_csv('train_set_jobs.csv', index=False)
    test_set.to_csv('test_set_jobs.csv', index=False)


if __name__ == '__main__':
    address = r'jobs_in_data.csv'  # Defining the file path for the CSV file
    df = pd.read_csv(address)  # Reading the data from CSV file into a DataFrame 'df'

    print(df.head(3))  # Printing the first 3 rows of the DataFrame to examine the data
    # print(df.info())  # Printing the DataFrame information, including column data types and non-null counts
    # print(df.isnull().sum())  # Checking for missing values in each column of the DataFrame

    # print(df.describe())  # - Provides summary statistics for numerical columns (count, mean, std, min, 25%, 50%, 75%, max)
    # print(df['job_title'].value_counts())  # - Counts the frequency of each unique value in a column


    # 3. Visualization: Plot histograms, box plots, scatter plots, etc., to visualize distributions and relationships
    # 4. Correlation analysis: Use df.corr() to compute pairwise correlation of columns, or visualize with heatmap
    # 5. Handling categorical variables: Explore unique values, check for misspellings or inconsistencies, consider encoding
    # 6. Outlier detection: Identify and investigate potential outliers in numerical variables
    # 7. Feature engineering: Create new features based on domain knowledge or relationships between existing features

    # bar_plot()
    load_data_to_train_and_test()

    train_data = pd.read_csv('train_set_jobs.csv')
    print(train_data.head(3))

