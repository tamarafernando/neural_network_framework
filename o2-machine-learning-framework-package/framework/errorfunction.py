import numpy as np


# ERROR Categorical Cross Entropy
def categorical_cross_entropy(h, y_one_hot):
    h = np.clip(h, a_min=0.000000001, a_max=None)
    J = ((y_one_hot * np.log(h)).sum(axis=1).mean() * -1) / len(h)
    return J


#ERROR Mean Squared Error
def mse(h, y):
    return np.mean((h - y)**2)