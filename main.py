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

if __name__ == '__main__':
    address = r'jobs_in_data.csv'  # Defining the file path for the CSV file
    df = pd.read_csv(address)  # Reading the data from CSV file into a DataFrame 'df'

    print(df.head(3))  # Printing the first 3 rows of the DataFrame to examine the data
    print(df.info())  # Printing the DataFrame information, including column data types and non-null counts
    print(df.isnull().sum())  # Checking for missing values in each column of the DataFrame

    print(df.describe()) #- Provides summary statistics for numerical columns (count, mean, std, min, 25%, 50%, 75%, max)
    print(df['job_title'].value_counts()) # - Counts the frequency of each unique value in a column
# 3. Visualization: Plot histograms, box plots, scatter plots, etc., to visualize distributions and relationships
# 4. Correlation analysis: Use df.corr() to compute pairwise correlation of columns, or visualize with heatmap
# 5. Handling categorical variables: Explore unique values, check for misspellings or inconsistencies, consider encoding
# 6. Outlier detection: Identify and investigate potential outliers in numerical variables
# 7. Feature engineering: Create new features based on domain knowledge or relationships between existing features

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
    sns.barplot(x='job_category', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by Job Category')
    plt.xlabel('Job Category')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='work_year', y='salary_in_usd', data=df, estimator=np.mean, ci=None)
    plt.title('Average Salary (USD) by Job Category')
    plt.xlabel('work_year')
    plt.ylabel('Average Salary (USD)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()