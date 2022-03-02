import numpy as np


def sig(z):
    return 1 / (1 + np.e ** -z)


def tanh(z):
    if type(z[0][0]) == np.complex128:
        z = z.real
    return np.tanh(z)


def derivative_tanh(activationlayer):
        return (1 - (activationlayer ** 2))


def derivative_sigmoid (activationlayer):
        return (activationlayer * (1 - activationlayer))


