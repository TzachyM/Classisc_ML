import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV


def outliers(train, v=2):
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
    return train


def fill_nan(df):
    # check for missing data
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(5))
    df.Cabin = [df.Cabin[i][0] if not pd.isnull(df.Cabin[i]) else '0' for i in range(df.shape[0])]
    index_NaN_age = list(df["Age"][df["Age"].isnull()].index)
    for i in index_NaN_age:
        age_med = df["Age"].median()
        age_pred = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"]) & (df['Parch'] == df.iloc[i]["Parch"]) & (
                    df['Pclass'] == df.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred):
            df['Age'].iloc[i] = age_pred
        else:
            df['Age'].iloc[i] = age_med
    df = df.apply(lambda x: x.fillna(x.median()) if is_numeric_dtype(x) else x.fillna(Counter(x).most_common(1)[0][0]))
    return df


def cat_order(df):
    col = ('Pclass', 'Age', 'family_size')
    for c in col:
        lbl = LabelEncoder()
        lbl.fit(list(df[c].values))
        df[c] = lbl.transform(list(df[c].values))
    return df


def visual(df):
    corrmat = df.corr()
    sns.heatmap(corrmat, square=True, annot=True)
    sns.catplot('Pclass', 'Survived', data=df, kind="bar")
    sns.catplot('Sex', 'Survived', data=df, kind="bar")
    sns.catplot('SibSp', 'Survived', data=df, kind="bar")
    sns.catplot('Parch', 'Survived', data=df, kind="bar")
    sns.catplot('Embarked', 'Survived', data=df, kind="bar")
    sns.FacetGrid(df, col='Survived').map(sns.distplot, "Age")


def feature_eng(df):
    df = df.iloc[:, 1:]
    # Age
    labels = ['0-9', '9-16', '17-30', '30-50', '51+']
    bins = [0, 10, 17, 31, 51, 90]
    df.Age = pd.cut(df.Age, bins, labels=labels, include_lowest=True)
    # Family size (sibsp+parch)
    df['family_size'] = df.SibSp + df.Parch
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    labels = ['Single', 'Small family', 'Medium Family', 'Big family']
    bins = [0, 1, 4, 7, 11]
    df.family_size = pd.cut(df.family_size, bins, labels=labels, include_lowest=True)
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['family_size'] != 'Single'] = 0
    sns.catplot('family_size', 'Survived', data=df, kind="bar")
    # Ticket
    Ticket = []
    for i in list(df.Ticket):
        if not i.isdigit():
            Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0][0])  # Take prefix
        else:
            Ticket.append("X")
    df.Ticket = Ticket
    # Names
    df.Name = df.Name.apply(lambda x: x.strip().split(" ")[1].replace(".", ""))
    df.Name = df.Name.apply(lambda x: 'Mr' if x == 'Mr' else 'Miss' if x == 'Miss' else 'Mrs' if x == 'Mrs' else 'Master' if x == 'Master' else 'Other')
    df = cat_order(df)
    df = pd.get_dummies(df)
    return df


def normal(x_train, x_test):
    min_max_scaler = Normalizer()
    train_scaled = min_max_scaler.fit_transform(x_train)
    test_scaled = min_max_scaler.transform(x_test)
    return pd.DataFrame(train_scaled, columns=x_train.columns), pd.DataFrame(test_scaled, columns=x_test.columns)


def cross_val(x_train, y_train):
    random_state = 9
    kfold = KFold(n_splits=5, random_state=random_state)
    classifiers = []
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state)))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    cv_results = []
    for classifier in classifiers:
        cv_results = cross_val_score(classifier, x_train, y_train, scoring="accuracy", cv=kfold, n_jobs=4).mean()
        print(f"classifier {classifier} accuracy is {cv_results}")
    GBC = hyper_param(kfold) #hyper parameters for the best classifier
    return GBC


def hyper_param(kfold):
    GBC = GradientBoostingClassifier()
    gb_param = {'loss': ["deviance"],
                'n_estimators': [100, 200, 300, 1000],
                'learning_rate': [0.2, 0.01, 0.02],
                'max_depth': [4, 8],
                'min_samples_leaf': [100, 150, 2],
                'max_features': [0.3, 0.1]}
    gsGBC = GridSearchCV(GBC, param_grid=gb_param, cv=kfold, scoring="accuracy", n_jobs=8, verbose=1)
    gsGBC.fit(x_train, y_train)
    GBC_best = gsGBC.best_estimator_
    print(GBC_best)
    return GBC_best


def rmsle_cv(model, train, y_train):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train)
    rmse = np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return rmse


class AveragingModels:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1).round()

    def score(self, y_pred, y):
        N = y.shape[0]
        return ((y == y_pred).sum() / N)
            
if __name__ == '__main__':
    #read data
    train = pd.read_csv(r'C:\Users\tzach\Dropbox\DS\Primrose\Exercises\Kaggle\titanic\train.csv')
    test = pd.read_csv(r'C:\Users\tzach\Dropbox\DS\Primrose\Exercises\Kaggle\titanic\test.csv')
    # removing NaN
    test = fill_nan(test)
    train = fill_nan(train)
    #visual(train)  # heat mat and plots
    train = outliers(train)  # removing outliers

    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    id_ = test.PassengerId
    df = feature_eng(df)
    y_train_org = df.iloc[:len(train), 0]
    x_train_org = df.iloc[:len(train), 1:]
    test = df.iloc[len(train):, 1:]
    x_train_org, test = normal(x_train_org, test)
    x_train, x_test, y_train, y_test = train_test_split(x_train_org, y_train_org)
    mode = cross_val(x_train, y_train) # finding the best algorithm and parameters to test
    
    GBoost = GradientBoostingClassifier(learning_rate=0.2, max_depth=8, max_features=0.1,
                           min_samples_leaf=150, n_estimators=1000)
    epochs = 100
    gb_test_score = []
    for i in range(epochs):
        GBoost.fit(x_train, y_train)
        gb_test_score.append(GBoost.score(x_test, y_test))
    print(f"Test accuarcy using only Gradient Boost {np.mean(gb_test_score)*100:.3f}%")
        
    AdaB = AdaBoostClassifier()
    Forest = RandomForestClassifier()
    av = AveragingModels([AdaB, GBoost, Forest])
    
    test_score = []
    for i in range(epochs):
        av.fit(x_train, y_train)
        y_pred = av.predict(x_test).astype(np.int64)
        test_score.append(av.score(y_pred, y_test))
    print(f"Accuracy of test data using average model is {np.mean(test_score)*100:.3f}%")
    
    av.fit(x_train, y_train)
    y_test_pred = av.predict(test).astype(np.int64)
