# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 09:04:52 2020

@author: Tzachy
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


def data_read():
    df = pd.read_csv(r'BankChurners.csv')
    df = df.rename(columns={
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1": 'NBC1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': 'NBC2'})
    y = df['Attrition_Flag']
    x = df.drop(['Attrition_Flag'], axis=1)

    return x, y


def visual_data(x, y):
    y = y.map({'Existing Customer': 0, 'Attrited Customer': 1})
    df = pd.concat([x, y], axis=1)

    print(f"The amount of NaN's is {df.isna().sum().sum()}")
    print(f"The amount of features with high skewness is {df.skew()[df.skew() > 1].count()}")
    corrmat = df.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corrmat, square=True, annot=True, ax=ax)
    sns.catplot('Customer_Age', 'Attrition_Flag', data=df, kind="bar")
    sns.catplot('Education_Level', 'Attrition_Flag', data=df, kind="bar")
    sns.catplot('Marital_Status', 'Attrition_Flag', data=df, kind="bar")
    sns.catplot('Income_Category', 'Attrition_Flag', data=df, kind="bar")
    return x, y


def feat_eng(x):
    print(f"The features with high skewness is {x.skew()[x.skew() > 1]}")
    x.drop(['CLIENTNUM', 'NBC2'], axis=1, inplace=True)
    x.Education_Level = x.Education_Level.map({'Uneducated': 0
                                                  , 'Graduate': 0, 'High School': 0, 'College': 1,
                                               'Post-Graduate': 2, 'Doctorate': 3})

    x.Income_Category = x.Income_Category.map({'Less than $40K': 0
                                                  , '$40K - $60K': 1, '$60K - $80K': 2,
                                               '$80K - $120K': 3, '$120K +': 4})
    x.Card_Category = x.Card_Category.map({'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3})
    x = x.apply(lambda x: x.fillna(x.median()) if is_numeric_dtype(x) else x.fillna(Counter(x).most_common(1)[0][0]))
    x = pd.get_dummies(x)
    return x


def normal(x_train, x_test):
    normal_ = StandardScaler()
    train_scaled = normal_.fit_transform(x_train)
    test_scaled = normal_.transform(x_test)
    return train_scaled, test_scaled


def outliers(x, y, v=2):
    train = pd.concat([x, y], axis=1)
    outliers_index = []
    for col in train.columns:
        if is_numeric_dtype(train[col]):
            q1 = np.percentile(train[col], 25)
            q3 = np.percentile(train[col], 75)
            iqr = q3 - q1
            step = 1.5 * iqr
            outliers_col_index = train[(train[col] < q1 - step) | (train[col] > q3 + step)].index
            outliers_index.extend(outliers_col_index)
    multi_outlier = Counter(outliers_index)
    multi_outlier = [k for k, f in multi_outlier.items() if f > v]
    train = train.drop(multi_outlier, axis=0).reset_index(drop=True)
    y = train['Attrition_Flag']
    x = train.drop(['Attrition_Flag'], axis=1)
    return x, y


x, y = data_read()
x, y = visual_data(x, y)
x = feat_eng(x)

x_train, x_test, y_train, y_test = train_test_split(x, y)
x_train, y_train = outliers(x_train, y_train)
x_train, x_test = normal(x_train, x_test)

model = AdaBoostClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(f'The accuracy score for our model is {score * 100}%')