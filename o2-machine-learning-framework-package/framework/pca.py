import numpy as np
import pandas as pd

def pcatraining(x, cutoff):
    features = x.T
    cov_matrix = np.cov(features)

    values, vectors = np.linalg.eig(cov_matrix)

    explained_variances = []
    for i in range(len(values)):
        if (np.sum(explained_variances) <= cutoff):
            explained_variances.append(values[i] / np.sum(values))
    numberofvectors = len(explained_variances)

    neededvectors = (vectors.T[:][:numberofvectors]).T
    x_pca = x.dot(neededvectors)
    if isinstance(x_pca, pd.DataFrame):
        x_pca = x_pca.to_numpy()

    return x_pca, neededvectors


def pcavalidation(xval, neededvectors):
    xval_pca = xval.dot(neededvectors)
    if isinstance(xval_pca, pd.DataFrame):
        xval_pca = xval_pca.to_numpy()

    return xval_pca
