import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

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



if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\tzach\Dropbox\DS\Primrose\Excercies\Kaggle\Digit\train.csv')
    test = pd.read_csv(r'C:\Users\tzach\Dropbox\DS\Primrose\Excercies\Kaggle\Digit\test.csv')
    label = df.iloc[:, 0]
    train = df.drop(['label'], axis=1)
    #plt.imshow(train.iloc[1].values.reshape(28, 28))
    #normal = StandardScaler()
    #normal.fit_transform(train)
    #x_train, x_test, y_train, y_test = train_test_split(train, label)
    #pca = PCA(n_components=10)
    #x_train = pca.fit_transform(x_train)
    #test = pca.transform(test)
    GBoost = GradientBoostingClassifier(learning_rate=0.2, max_depth=8, max_features=0.1,
                                        min_samples_leaf=150, n_estimators=1000)
    SV = SVC()
    Forest = RandomForestClassifier(n_jobs=8)
    #av = AveragingModels([SV, GBoost, Forest])
    GBoost.fit(train, label)
    100101
    y_pred = GBoost.predict(test).astype(np.int64)

    # submission
    submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': y_pred})
    submission.to_csv(r'C:\Users\tzach\Dropbox\DS\Primrose\Excercies\Kaggle\Digit\submission.csv', index=False)