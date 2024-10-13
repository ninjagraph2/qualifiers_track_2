import numpy as np

from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

class LogRegPCA:
    def __init__(self, pca=True, penalty='l2', C=1.0, solver='lbfgs', max_iter=500):
        self.pca = PCA(n_components=0.98) if pca else None
        self.scaler = StandardScaler()
        self.model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter, random_state=42)

    def model_training(self, x, y):
        x = self.preprocess(x)
        x = self.scaler.fit_transform(x)

        if self.pca is not None:
            x = self.pca.fit_transform(x)

        self.model.fit(x, y.ravel())

        acc = self.model.score(x, y)
        print('Accuracy on train:', round(acc, 3))

        return acc
    def model_predict(self, x):
        x = self.preprocess(x)
        x = self.scaler.transform(x)

        if self.pca is not None:
            x = self.pca.transform(x)

        return self.model.predict(x)

    def model_testing(self, x, y):
        y_pred = self.model_predict(x)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        print('Accuracy on test:', round(acc, 3))
        print('F1 score on test:', round(f1, 3))
        cm = confusion_matrix(y, y_pred)

        return cm, acc, f1

    def preprocess(self, x):
        vecs = zscore(x, axis=0)

        for i in vecs:
            np.fill_diagonal(i, 0)

        vecs = vecs.reshape((x.shape[0], -1))

        return vecs