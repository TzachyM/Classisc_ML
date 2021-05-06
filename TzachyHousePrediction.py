import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error, r2_score

import lightgbm as lgb


def import_data():
    train = pd.read_csv(r'C:\Users\tzach\Dropbox\DS\Primrose\Exercises\Kaggle\House Price\train.csv')
    test = pd.read_csv(r'C:\Users\tzach\Dropbox\DS\Primrose\Exercises\Kaggle\House Price\test.csv')
    return train, test

def data_prep():
    train, test = import_data()
    #Outlier removel
    train = outlier_remove(train)
    #Divding data
    id_test = test.Id
    y_train = train.SalePrice
    df = pd.concat([train, test]).reset_index(drop=True).drop(['Id', 'SalePrice'], axis=1)
    # Visual data
    #visual(train)
    #Feature engineering
    df = feat_engineering(df)
    train = df.iloc[:train.shape[0], :]
    test = df.iloc[train.shape[0]:, :]
    return train, test, y_train, id_test

def feat_engineering(df):
    #Fill NaN
    df = fill_na(df)
    #Features
    df = binary_fix(df, 'Condition1', 'Norm')
    df = binary_fix(df, 'Condition2', 'Norm')
    df['Condition'] = df.Condition1 + df.Condition2
    labels = [0, 1, 2, 3]
    bins = [0, 1900, 1950, 2000, 2020]
    df.YearBuilt = pd.cut(df.YearBuilt, bins, labels=labels, include_lowest=True)
    labels = [0, 1, 2, 3, 4]
    bins = [0, 1950, 1970, 1990, 2000, 2020]
    #df.YearRemodAdd = pd.cut(df.YearRemodAdd, bins, labels=labels, include_lowest=True)  # maybe remove
    df = binary_fix(df, 'RoofMatl', 'WdShngl')
    labels = [0, 1, 2]
    bins = [0, 1, 400, 2000]
    #df.MasVnrArea = pd.cut(df.MasVnrArea, bins, labels=labels, include_lowest=True) #maybe leave it numerical
    value_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Na': 0, 'Av': 3, 'Mn': 2, 'No': 1, 'GLQ': 6, 'ALQ': 5,
                  'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1}
    df.ExterQual = df.ExterQual.map(value_dict)
    df.ExterCond = df.ExterCond.map(value_dict)
    df.BsmtQual = df.BsmtQual.map(value_dict)
    df.BsmtCond = df.BsmtCond.map(value_dict)
    df.BsmtExposure = df.BsmtExposure.map(value_dict)
    df.BsmtFinType1 = df.BsmtFinType1.map(value_dict)
    labels = [0, 1, 2, 3, 4]
    bins = [0, 1, 500, 1000, 1500, 3000]
    df.BsmtFinSF1 = pd.cut(df.BsmtFinSF1, bins, labels=labels, include_lowest=True)
    df.BsmtUnfSF = pd.cut(df.BsmtUnfSF, bins, labels=labels, include_lowest=True)
    df.loc[(df.Heating == 'GasA') | (df.Heating == 'GasW'), 'Heating'] = 1
    df.loc[df.Heating != 1, 'Heating'] = 0
    df.Heating = df.Heating.astype(int)
    df.HeatingQC = df.HeatingQC.map(value_dict)
    df['TotalFlr'] = df['1stFlrSF'] + df['2ndFlrSF']
    df['2ndFlrSF'] = pd.cut(df['2ndFlrSF'], bins, labels=labels, include_lowest=True)
    df = binary_fix(df, 'KitchenAbvGr', 1)
    df.KitchenQual = df.KitchenQual.map(value_dict)
    df.TotRmsAbvGrd = df.TotRmsAbvGrd.map({1: 1, 2: 2, 3: 4, 4: 4, 5: 4, 6: 6, 7: 7, 8: 8, 9: 9, 10: 11, 11: 11,
                                           12: 11, 13: 13, 14: 14, 15: 15})
    df.Functional = binary_fix(df, 'Functional', 'Maj2')
    df.FireplaceQu = df.FireplaceQu.map(value_dict)
    df.GarageFinish = df.GarageFinish.map({'Na': 0, 'Fin': 3, 'RFn': 2, 'Unf': 1})
    df.GarageQual = df.GarageQual.map(value_dict)
    df.GarageCond = df.GarageCond.map(value_dict)
    df.PavedDrive = df.PavedDrive.map({'Y': 2, 'P': 1, 'N': 0})
    labels = [0, 1, 2, 3, 4, 5]
    bins = [0, 1, 100, 200, 300, 400, 1000]
    df.WoodDeckSF = pd.cut(df.WoodDeckSF, bins, labels=labels, include_lowest=True)
    labels = [0, 1, 2, 3]
    bins = [0, 1, 100, 200, 1000]
    df.OpenPorchSF = pd.cut(df.OpenPorchSF, bins, labels=labels, include_lowest=True)
    #Columns drop
    df.drop(['Condition1', 'Condition2', 'Utilities','BsmtFinSF2', 'BsmtFinType2', 'LowQualFinSF','BsmtFullBath',
     'BsmtHalfBath', 'HalfBath', 'GarageYrBlt', 'GarageArea', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 'PoolArea',
     'PoolQC', 'Fence', 'MiscVal', 'MoSold', 'YrSold', 'MSSubClass', 'Foundation', 'RoofStyle'], axis=1, inplace=True)  # 'MSSubClass','Foundation','RoofStyle'
    #Skewness fix
    #print("Skewness before fix:", df.skew().mean())
    df = df.apply(lambda x: np.log1p(x) if is_numeric_dtype(x) and np.abs(x.skew())>2 else x)
    #print("Skewness after fix:", df.skew().mean())
    df = pd.get_dummies(df)
    return df

def normal(train, test):
    norm = MinMaxScaler()
    train = norm.fit_transform(train)
    test = norm.transform(test)
    return train, test

def binary_fix(df, col, value):
    # df.loc[df[col] == value, col] = 0
    # df.loc[df[col] != 0, col] = 1
    # df[col] = df[col].astype(int)
    return df

def fill_na(df):
    df['Alley'] = df['Alley'].fillna('No')
    df['LotFrontage'] = df['LotFrontage'].fillna(0)
    for name in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'FireplaceQu', 'GarageType', 'GarageFinish',
                 'GarageQual','GarageCond', 'MiscFeature']:
        df[name] = df[name].fillna('Na')
    df.fillna(df.mode().iloc[0], inplace=True)
    return df

def outlier_remove(df, iqr_step=1.3, n=3):
    outlier_index = []
    for col in df.columns:
        if(is_numeric_dtype(df[col])):
            q1 = np.percentile(df[col], 25)
            q3 = np.percentile(df[col], 75)
            iqr = q3-q1
            step = iqr_step*iqr
            outlier_col = df[(df[col]<q1-step) | (df[col]>q3+step)].index
            outlier_index.extend(outlier_col)
    outlier_indices = Counter(outlier_index)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    df.drop(multiple_outliers, inplace=True)
    return df

def visual(df):
    # view year build with salePrice
    df.plot.scatter(x='YearBuilt', y='SalePrice', xlim=(1880, 2020))
    # view RoofStyle with salePrice
    plt.figure()
    sns.boxplot(x='RoofStyle', y="SalePrice", data=df)
    # view RoofMatl  with salePrice
    plt.figure()
    sns.boxplot(x='RoofMatl', y="SalePrice", data=df)
    # view MasVnrArea with salePrice
    df.plot.scatter(x='MasVnrArea', y='SalePrice')
    # view BsmtFinSF1 with salePrice
    df.plot.scatter(x='BsmtFinSF1', y='SalePrice')
    # view TotalBsmtSF with salePrice
    df.plot.scatter(x='TotalBsmtSF', y='SalePrice')

def rmsle_cv(model, train, y_train):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train)
    r2 = np.sqrt(-cross_val_score(model, train, y_train, scoring="r2", cv = kf))
    return r2


class AveragingModels:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([ model.predict(X) for model in self.models ])
        return np.mean(predictions, axis=1)
    
    
def cross_val_models(x_train, y_train, cv_param=5):
    ABR = AdaBoostRegressor()
    GBR = GradientBoostingRegressor()
    RF = RandomForestRegressor()
    Las = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))   # Lasso is better used with RobustScaler
                                                                               # and pipeline, thus we gave him his own
                                                                               # parameters.

    #best_est = hyperparam(ABR, GBR, RF, x_train, y_train)  # this part take some time, according to your hardware, so we added the
    # hyperparameters manually after running the code, to run it, just remove the '#' from the start of the row
    # and add the best_est according to the model.
    #print(best_est)
    GBR = GradientBoostingRegressor(learning_rate=0.2, max_depth=8, max_features='sqrt',
                          min_samples_leaf=0.1,
                          min_samples_split=0.13636363636363638,
                          n_estimators=10, subsample=0.95)
    ABR = AdaBoostRegressor(learning_rate=0.3, loss='exponential', n_estimators=100)
    RF = RandomForestRegressor(max_depth=8, min_samples_leaf=2, n_estimators=800)
    models = [ABR, GBR, RF, Las]
    for model in models:    # Cross validation of the train data with the different models
        cv_results = -cross_val_score(model, x_train, y_train, cv=cv_param, scoring='r2')
        mean_cv = cv_results.mean()
        model_name = type(model).__name__
        if model_name == 'Pipeline':
            model_name = 'Lasso'
        print(f'The r2 for {model_name} is {mean_cv}')
    return models

# Hyperparameters scan using GridSearchCV (Note, this process take a couple of minutes even with an 8 core computer)

def hyperparam(ABR, GBR, RF, x_train, y_train):
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
        clf = GridSearchCV(model, param, n_jobs=-1, scoring='neg_mean_absolute_error')
        clf.fit(x_train, y_train)
        print(clf.best_estimator_)
        best_est.append(clf)
    return best_est

# Stacking the models with a final regressor to achieve a better MSE:
## Stacking allows us to use each individual estimator by using their output as input of a final estimator.

def stacking(models, x_train, x_test, y_train, test):
    estimators_ = []
    for model in models:
        estimators_.append((str(model), model))
    stack = StackingRegressor(estimators=estimators_, final_estimator=RandomForestRegressor(n_estimators=10,
                                                                                            random_state = 42))
    stack.fit(x_train, y_train)
    print("R2 Score for stacking models with train data",stack.score(x_train, y_train))
    y_pred = stack.predict(x_test)
    print("R2 Score for stacking models with test data",stack.score(x_test, y_pred))
    y_pred_stack = stack.predict(test)
    return y_pred_stack


def average_stacking(x_train, y_train, x_test, y_test):
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)

    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    score = rmsle_cv(GBoost, x_train, y_train)
    print(f"Gradient Boosting R2 score with train data: {score.mean():.4f} | std: ({score.std():.4f})\n")
    score = rmsle_cv(model_lgb, x_train, y_train)
    print("LGBM R2 score with train data: {score.mean():.4f} ({score.std():.4f})\n")

    av = AveragingModels([model_lgb, GBoost, lasso])

    av.fit(x_train, y_train)
    
    y_pred = av.predict(x_test)
    avg_score = r2_score(y_test, y_pred)
    print("Average score of LGBM, Gradient boosting and Lasso with test data", avg_score)
    return y_pred


if __name__ == "__main__":

    train, test, y_train, id_test = data_prep()
    y_train = np.log1p(y_train) #Fix skew data
    train, test = normal(train, test)    # Normalize the data
    x_train, x_test, y_train, y_test = train_test_split(train, y_train)
    y_pred_average = average_stacking(x_train, y_train, x_test, y_test)

    models = cross_val_models(x_train, y_train, cv_param=6)
    y_pred = stacking(models, x_train, x_test, y_train, test)
    y_pred_stack = np.exp(y_pred) #cancel the log we implmented


# =============================================================================
#     submission = pd.DataFrame({'Id': id_test, 'SalePrice': y_pred_stack})
# 
#     submission.to_csv(r'C:\Users\tzach\Dropbox\DS\Primrose\Exercises\Kaggle\House Price\test.submission.csv',
#                       index=False)
# 
#     submission = pd.read_csv(r'C:\Users\tzach\Dropbox\DS\Primrose\Exercises\Kaggle\House Price\test.submission.csv')
# 
#     print(submission)
# =============================================================================

