# Imported Libraries:

import numpy as np  # linear algebra
import pandas as pd  # data processing
import seaborn as sns  # visual data
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor  # ML models
from sklearn.linear_model import Lasso  # ML model
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor  # Stacking for linear regression


# Reading data from CSV:
## As you can from looking at the data, we have a big flaw in the data. The first rows needs to be removed because of
## the needless heading. Thus, we are changing the rows to have a normal dataframe

def data_read():
    df = pd.read_csv(r'C:\Users\tzach\Dropbox\DS\Primrose\Excercies\Kaggle\CancerCostUS\Cancer.csv', header=None)
    print(df.info(verbose=True, null_counts=True))
    col_index = df.iloc[3, :].values  # saving the features names for later.
    df = df.iloc[4:, :].reset_index(drop=True)
    df.columns = col_index  # returning the features names
    return df

# Looking at the data and the unique values in each row as the data itself is mostly made of repeated data:

def data_inquiry(df):
    print(f'the data has {df.isna().sum().sum()} NaNs')
    print(df.info)  # viewing the data summary and realize that something has gone horribly wrong
    for col in df.iloc[:, :-3]:  # running over the columns to check for repeated values, minus the last 3 numeric ones.
        print("Column: ", col, df[col].unique())

# Feature engineering of the different columns:

def feature_engineer(df):
    df.drop(['Last Year of Life Cost', 'Continuing Phase Cost', 'Initial Year After Diagnosis Cost', 'Age'], axis=1,
            inplace=True)  #Droping columns with repeated
    df.rename(columns={'Incidence and Survival Assumptions': 'Incidence+Survival',
                         'Annual Cost Increase (applied to initial and last phases)': 'Cost'}, inplace=True)
    df['Sex'] = df['Sex'].map({'Both sexes': 0, 'Females': 1, 'Males': 2})  # Categorized data with no order
    df['Cost'] = df['Cost'].map({'0%': 0, '2%': 1, '5%': 2}).astype('category')  # Changing the data to categories with an order
    df['Total Costs'] = df['Total Costs'].astype('float64')  # Changing from an object type.
    ax = sns.boxplot(x="Year", y="Total Costs", data=df)  # Visual the Year and Total Costs after changing the data type
    print(f"Data skewness: {df.skew()}")  # Skewness check
    x = df.iloc[:, :-1]
    x = pd.get_dummies(x)   # Convert categorical variable into indicator variables
    y = df.iloc[:, -1]
    return x, y, df

def visual(df):
    sns.catplot(x='Cancer Site', y='Total Costs', color='#FB8861',kind="box", data=df, legend_out=False, height=10, aspect=2)
    sns.catplot(x='Sex', y='Total Costs', color='r',kind="violin", data=df)
    sns.catplot(x='Incidence+Survival', y='Total Costs',kind="box", color='b', data=df, height=10, aspect=2 )
    sns.catplot(x='Cost', y='Total Costs', kind="box", color='g', data=df)

# Dividing the data into the train and test:

def train_test_split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    y_train = np.log1p(y_train)  # Using log to fix the skewness of the label (log1p is to help vs negative values)
    print(f"Data skewness after log: {df.skew()}")
    return x_train, x_test, y_train, y_test

# Cross validation of different models:

def cross_val_models(x_train, y_train, cv_param=5):
    ABR = AdaBoostRegressor()
    GBR = GradientBoostingRegressor()
    RF = RandomForestRegressor()
    Las = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))   # Lasso is better used with RobustScaler
                                                                               # and pipeline, thus we gave him his own
                                                                               # parameters.

    best_est = hyperparam(ABR, GBR, RF, x_train, y_train)  # this part take some time, according to your hardware, so we added the
    # hyperparameters manually after running the code, to run it, just remove the '#' from the start of the row
    # and add the best_est according to the model.

    GBR = GradientBoostingRegressor()   # Surprisingly we got better results using the default parameters
    ABR = AdaBoostRegressor(learning_rate=1, loss='square', n_estimators=100)
    RF = RandomForestRegressor(max_depth=8, n_estimators=600)
    models = [ABR, GBR, RF, Las]
    for model in models:    # Cross validation of the train data with the different models
        cv_results = -cross_val_score(model, x_train, y_train, cv=cv_param, scoring='neg_mean_squared_error')
        mean_cv = cv_results.mean()
        model_name = type(model).__name__
        if model_name == 'Pipeline':
            model_name = 'Lasso'
        print(f'The mean_squared_error for {model_name} is {mean_cv}')
    return models

#Hyperparameters scan using GridSearchCV (Note, this process take a couple of minutes even with an 8 core computer)

def hyperparam(ABR, GBR, RF):
    RF_param = {
        'max_depth': [4, 6, 8],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800]}
    GB_param = {
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        "min_samples_split": np.linspace(0.1, 0.5, 12),
        "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        "max_depth": [3, 5, 8],
        "max_features": ["log2", "sqrt"],
        "criterion": ["friedman_mse", "mae"],
        "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        "n_estimators": [10]}
    AB_param = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
        'loss': ['linear', 'square', 'exponential']}
    param_list = [RF_param, GB_param, AB_param]
    model_list = [RF, GBR, ABR]
    best_est = []
    for param, model in zip(param_list, model_list):
        clf = GridSearchCV(model, param, n_jobs=-1, scoring='neg_mean_squared_error')
        clf.fit(x_train, y_train)
        print(clf.best_estimator_)
        best_est.append(clf)
    return best_est

# Stacking the models with a final regressor to achieve a better MSE:
## Stacking allows us to use each individual estimator by using their output as input of a final estimator.

def stacking(models, x_train, x_test, y_train, y_test):
    estimators_ = []
    for model in models:
        estimators_.append((str(model), model))
    stack = StackingRegressor(estimators=estimators_, final_estimator=RandomForestRegressor(n_estimators=10,
                                                                                            random_state = 42))
    stack.fit(x_train, y_train)
    y_pred = stack.predict(x_test)
    mse = np.square(y_pred-np.log1p(y_test)).mean()  # Final MSE calculation while remembering to adapt the y_test with
                                                     # the log, like we did with the y_train
    return mse


if __name__ == '__main__':

    df = data_read()
    data_inquiry(df)
    x, y, df = feature_engineer(df)
    visual(df)
    x_train, x_test, y_train, y_test = train_test_split_data(x, y)
    models = cross_val_models(x_train, y_train, cv_param=6)
    mse = stacking(models, x_train, x_test, y_train, y_test)
    print(f'The final MSE of the stacked models compared with the y_test is {mse}')