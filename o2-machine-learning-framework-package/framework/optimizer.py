import numpy as np
from framework import activationfunction, errorfunction
import matplotlib.pyplot as plt


# Neural Network
class NeuralNetwork():

    def __init__(self, x, classificationdictionary, activationfunctionname, errorfunctionname, architecturearray,
                 regularizationlambda):

        if errorfunctionname == "MSE" and architecturearray != [0]:
            print(
                "mse starts a linear regression, therefore your architecture array should be [0], please adapt your parameters!")
            return None
        self.ACTIVATION_FUNCTION = {
            "sigmoid": activationfunction.sig,
            "tanh": activationfunction.tanh,
            "regression": None
        }
        self.ACTIVATION_DERIVATIVE = {
            "sigmoid": activationfunction.derivative_sigmoid,
            "tanh": activationfunction.derivative_tanh,
            "regression": None
        }
        self.ERRORFUNCTION = {
            "CrossEntropy": errorfunction.categorical_cross_entropy,
            "MSE": errorfunction.mse
        }
        if classificationdictionary == "regression":
            self.outputneurons = 1
        else:
            self.outputneurons = len(classificationdictionary)
        self.activationfunctionname = activationfunctionname
        self.errorfunctionname = errorfunctionname
        self.thetasarray = []
        self.biasarray = []
        self.samplesize = x.shape[0]
        self.featuresize = x.shape[1]
        self.regularizationlambda = regularizationlambda
        if classificationdictionary == 1:
            self.classificationdictionary = 1
        self.classificationdictionary = classificationdictionary

        self.inizialize_thetas(architecturearray)

    def inizialize_thetas(self, architecturearray):
        if architecturearray == [0]:
            self.thetasarray.append(np.random.rand(1, self.featuresize) - 0.5)
            self.biasarray.append(np.random.rand(1, 1) - 0.5)
        else:
            self.thetasarray.append(np.random.rand(architecturearray[0], self.featuresize) - 0.5)
            self.biasarray.append(np.random.rand(architecturearray[0], 1) - 0.5)
            for i in range(len(architecturearray) - 1):
                self.thetasarray.append(np.random.rand(architecturearray[i + 1], architecturearray[i]) - 0.5)
                self.biasarray.append(np.random.rand(architecturearray[i + 1], 1) - 0.5)
            self.thetasarray.append(np.random.rand(self.outputneurons, architecturearray[-1]) - 0.5)
            self.biasarray.append(np.random.rand(self.outputneurons, 1) - 0.5)

            self.thetasarray = np.array(self.thetasarray, dtype=object)
            self.biasarray = np.array(self.biasarray, dtype=object)

    def gradientdescent(self, alpha, iterations, x, y):
        if self.errorfunctionname == "MSE":
            self.gradientdescent_linear(alpha, iterations, x, y)
        else:
            self.gradientdescent_classification(alpha, iterations, x, y)

    ###neu
    def gradientdescent_linear(self, alpha, iterations, x, y):

        error_history = []
        for i in range(iterations):
            h = x @ self.thetasarray[0].T + self.biasarray[0]
            err = self.error_function(h, y)  # err = MSE
            error_history.append(err)
            gradients = (h - y) * (1 / self.samplesize)  * self.thetasarray[0]  # mse derivative
            self.thetasarray[0] = self.thetasarray[0] - gradients * alpha
        self.visualize_error_history(error_history)

    ###ende neu

    def gradientdescent_classification(self, alpha, iterations, x, y):

        yoh = self.onehot(y, self.outputneurons)

        error_history = []

        # Forward Propagation
        # initialisieren eines array das erstmal eindimensional ist aber an jedes seiner stellen ein Obj packen kann, daher 3 Dimensional werden kann
        activationarray = np.zeros(len(self.thetasarray), dtype=object)
        for cycle in (range(iterations)):
            for i in range(len(activationarray)):
                if (i == 0):
                    test = x @ self.thetasarray[i].T + self.biasarray[i].T
                    activationarray[i] = self.activation_function(test)
                elif (i == len(activationarray) - 1):  # für den letzten Fall (da Softmax selbst aktiviert)
                    activationarray[i] = activationarray[i - 1] @ self.thetasarray[i].T + self.biasarray[i].T
                else:
                    activationarray[i] = self.activation_function(
                        activationarray[i - 1] @ self.thetasarray[i].T + self.biasarray[i].T)

            # softmaxen
            s = (self.softmax(activationarray[-1]))  # nach dem Softmaxen

            yoh_pred = self.onehot(np.argmax(s, axis=1), self.outputneurons)

            err = self.error_function(s, yoh)
            err = self.regularize_error(err)
            error_history.append(err)

            # Backpropagation:
            deltaneurons = np.zeros(len(activationarray), dtype=object)
            deltathetas = np.zeros(len(self.thetasarray), dtype=object)

            for j in reversed(range(len(deltaneurons))):

                if (j == len(deltaneurons) - 1):
                    deltaneurons[j] = (s - yoh)
                    deltathetas[j] = deltaneurons[j].T @ activationarray[j - 1] / np.shape(deltaneurons[j].T)[1]

                if (j != 0 and j != len(deltaneurons) - 1):
                    deltaneurons[j] = (deltaneurons[j + 1] @ self.thetasarray[j + 1]) * self.activation_derivative(
                        activationarray[j])

                    deltathetas[j] = (deltaneurons[j].T @ activationarray[j - 1]) / np.shape(deltaneurons[j].T)[1]
                if (j == 0):
                    deltaneurons[j] = (deltaneurons[j + 1] @ self.thetasarray[j + 1]) * self.activation_derivative(
                        activationarray[j])
                    deltathetas[j] = (deltaneurons[j].T @ x) / np.shape(deltaneurons[j].T)[1]

            for layer in range(len(activationarray)):
                self.biasarray[layer] = self.biasarray[layer] - (
                    self.regularize_bias(deltaneurons[layer], layer, alpha))
                self.thetasarray[layer] = self.thetasarray[layer] - (
                    self.regularize_thetas(deltathetas[layer], layer, alpha))

        accuracy = self.accuracy(s, yoh)
        print("The error for the trainings set is:", err, "/n the accuracy is:", accuracy)
        self.f1_scores_classes(self.classificationdictionary, yoh, yoh_pred)
        self.visualize_error_history(error_history)

    def regularize_bias(self, deltaneuronsOfLayer, layer, alpha):
        if self.regularizationlambda == 0:
            return deltaneuronsOfLayer.mean() * alpha
        else:
            return deltaneuronsOfLayer.mean() + (((self.regularizationlambda / self.samplesize) * self.biasarray[layer])*alpha)

    def regularize_thetas(self, deltathetasOfLayer, layer, alpha):
        if self.regularizationlambda == 0:
            return deltathetasOfLayer * alpha
        else:
            return deltathetasOfLayer + (((self.regularizationlambda / self.samplesize) * self.thetasarray[layer]) * alpha)

    def regularize_error(self, j):
        if self.regularizationlambda == 0:
            return j
        else:
            return (j - ((self.regularizationlambda / (2 * self.samplesize)) * np.sum(self.thetasarray[-1] ** 2)))

    def f1_scores_classes (self, classificationdictionary, yoh, yoh_prediction):
        f1_ofAllClasses = np.zeros((self.outputneurons))
        for classes in range(yoh.shape[1]):
            f1_ofAllClasses[classes] = self.f1_score(yoh_prediction[0:yoh.shape[0], classes], yoh[0:yoh.shape[0], classes])

        print("The mean F1 Score is:", np.mean(f1_ofAllClasses))
        for classes in range(yoh.shape[1]):
            print("F1 Score for ", classificationdictionary[classes], " : ", f1_ofAllClasses[classes])
            print("____________")
        return f1_ofAllClasses

    def f1_score(self, h, y):
        h = np.round(h)

        true_positives = (h == 1) & (y == 1)
        false_positives = (h == 1) & (y == 0)
        false_negatives = (h == 0) & (y == 1)

        if ((np.sum(true_positives) == 0 and np.sum(false_positives)) or (
                np.sum(true_positives) == 0 and np.sum(false_negatives))):
            return float('-inf')

        precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
        recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def accuracy(self, s, yoh):
        return (s.argmax(axis=1) == yoh.argmax(axis=1)).mean()

    def activation_function(self, activationlayer):
        return self.ACTIVATION_FUNCTION[self.activationfunctionname](activationlayer)

    def activation_derivative(self, activationlayer):
        return self.ACTIVATION_DERIVATIVE[self.activationfunctionname](activationlayer)

    def error_function(self, h, y):
        return self.ERRORFUNCTION[self.errorfunctionname](h, y)

    def onehot(self, labels, numberofclasses):
        yoh = np.identity(numberofclasses)[labels]
        return yoh

    def softmax(self, o):
        return (np.exp(o)) / (np.exp(o)).sum(axis=1).reshape(-1, 1)

    # Error-History darstellen
    def visualize_error_history(self, error_history):
        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(error_history, label="Cross Entropy")

        ax.set_xlabel("iterations")
        ax.set_ylabel("Error")
        ax.set_title("Learning Curve")
        fig.legend()


class NetworkValidation():
    def __init__(self, thetas, bias, activationfunctionname):
        self.ACTIVATION_FUNCTION = {
            "sigmoid": activationfunction.sig,
            "tanh": activationfunction.tanh}
        self.thetas = thetas
        self.bias = bias
        self.activationfunctionname = activationfunctionname

    def prediction(self, x):

        activationarray = np.zeros(len(self.thetas), dtype=object)
        for i in range(len(activationarray)):
            if i == 0:
                activationarray[i] = self.activation_function(x @ self.thetas[i].T + self.bias[i].T)
            elif i == len(activationarray) - 1:  # für den letzten Fall (da Softmax selbst aktiviert)
                activationarray[i] = activationarray[i - 1] @ self.thetas[i].T + self.bias[i].T
            else:
                activationarray[i] = self.activation_function(activationarray[i - 1] @ self.thetas[i].T + self.bias[i].T)
        softmax = self.softmax(activationarray[-1])

        return softmax

    def activation_function(self, activationlayer):
        return self.ACTIVATION_FUNCTION[self.activationfunctionname](activationlayer)

    def softmax(self, o):
        return (np.exp(o)) / (np.exp(o)).sum(axis=1).reshape(-1, 1)



