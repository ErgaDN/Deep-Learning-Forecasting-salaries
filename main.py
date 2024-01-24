import keras
import sklearn.model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf
from xgboost import XGBClassifier
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

tf.disable_v2_behavior()
import numpy as np
import pandas as pd

if __name__ == '__main__':
    address = r'jobs_in_data.csv'
    df = pd.read_csv(address)

    #   Change features with words  to numbers
    labelencoder = LabelEncoder()

    df['job_title'] = labelencoder.fit_transform(df['job_title'])
    df['job_category'] = labelencoder.fit_transform(df['job_category'])
    df['salary_currency'] = labelencoder.fit_transform(df['salary_currency'])
    df['employee_residence'] = labelencoder.fit_transform(df['employee_residence'])
    df['experience_level'] = labelencoder.fit_transform(df['experience_level'])
    df['employment_type'] = labelencoder.fit_transform(df['employment_type'])
    df['work_setting'] = labelencoder.fit_transform(df['work_setting'])
    df['company_location'] = labelencoder.fit_transform(df['company_location'])
    df['company_size'] = labelencoder.fit_transform(df['company_size'])



