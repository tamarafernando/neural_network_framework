# O2. Machine Learning Framework Package

You are currently in the repository of "O2. Machine Learning Framework Package" of the Group 3 Machine Learning project in the winter semester 2020/2021. The repository provides a Python package for artificial neural networks. This package serves as the basis of all our trained neural networks. It contains one main package:
- [`framework`](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/lectures/machine-learning/student-material/ws20/Team-03/o2-machine-learning-framework-package/-/tree/master/framework) package


## Framework structure

Each subitem here is a separate script.

For a detailed explanation of each script, check out the [Wiki](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/lectures/machine-learning/student-material/ws20/Team-03/o2-machine-learning-framework-package/-/wikis/home).


### Activation functions `activationfunction.py`

- Sigmoid function
- Tanh function


### Error functions in `errorfunction.py`

- Categorical Cross Entropy (with Softmax)
- Mean Squared Error


### Optimizer in `optimizer.py`

- Gradient Descent without Momentum


### PCA in `pca.py`

- PCA for training and validation


### Scalers in `scaler.py`

- Normalizing 
- Standardizing

## Getting started

The framework gives you the possibility to solve classification problems and regression problems. In addition, there are two different ways to prepare data: scaling and performing PCA. Have a look in the Wiki [pca.py](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/lectures/machine-learning/student-material/ws20/Team-03/o2-machine-learning-framework-package/-/wikis/Home/pca.py) to learn how to apply a PCA. Check the Wiki [scaler.py](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/lectures/machine-learning/student-material/ws20/Team-03/o2-machine-learning-framework-package/-/wikis/Home/scaler.py) to leran how to scale your data. Check the whole [Wiki](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/lectures/machine-learning/student-material/ws20/Team-03/o2-machine-learning-framework-package/-/wikis/Home) for more information about all scripts of our framework.

### Classification problems
For classification problems the use of a neural network is recommended. You can do so, following two steps.

**First step: initializing the network**

In the first step, the network is initialized with the desired parameters. It returns the desired network, with which you can now continue working.
`optimizer.NeuralNetwork(x, classificationdictionary, activationfunctionname, errorfunctionname, architecturearray, regularizationlambda)`

| Parameter | Input | |
| ------ | ------ | ------ |
| x | ndarray | Training data set: An array where each row represents a sample and contains all features of the training dataset entered into it (each feature=one column. |
| classificationdictionary | dictionary | A dictionary that contains all classes to be learned. The value of the dictionary has to be an integer. |
| activationfunctionname | {"sigmoid", "tanh"} | Which activation function should be used for your hidden layers? Choose between sigmoid function (input "sigmoid") or tangens hyperbolicus (input "tanh"). |
| errorfunctionname | {"CrossEntropy"} | The choice of error function determines whether you want to solve a regression problem or a classification problem. You should choose the cross entropy function (input "CrossEntropy").  The MSE is intended for regression problems. |
| architecturearray | ndarray | Desired architecture of the neural network. In the array entered, each column represents a layer of the neural network. The number determines the number of neurons in the layer. For example `architecturearray=[12,12]` creates a network with two layers of 12 neurons each. |
| regularizationlambda | float | Entered value determines the size of lambda with which regularization is performed. If `regularizationlambda=0` then no regularization is applied to the network. |


**Second step: train your data**

Call this function via the previously created object of the network class with the desired parameters.

`optimizer.gradientdescent(alpha, iterations, x, y)`
| Parameter | Input | |
| ------ | ------ | ------ |
| alpha | float | Desired alpha. |
| iterations| int | Desired number of iterations. |
| x | ndarray | Training data set: An array where each row represents a sample and contains all features of the training dataset entered into it (each feature=one column. |
| y | ndarray | An array with as many rows as samples, containing the correct label for each sample of the data (ground truth). |


At the end of the function the accuracy over all classes and the F1 score for each class individually are output to the console. Trained weights and bias can be accessed too:

| Command | Type | |
| ------ | ------ | ------ |
| netz.thetasarray | ndarray | Returns the trained weights. |
| netz.biasarray | ndarray | Returns the bias. |



### Regression problems
For regression problems the use of a neural network is recommended. You can do so, following two steps.

**First step: initializing the network**

In the first step, the network is initialized with the desired parameters. It returns the desired network, with which you can now continue working.
`optimizer.NeuralNetwork(x, classificationdictionary, activationfunctionname, errorfunctionname, architecturearray, regularizationlambda)`

| Parameter | Input | |
| ------ | ------ | ------ |
| x | ndarray | Training data set: An array where each row represents a sample and contains all features of the training dataset entered into it (each feature=one column. |
| classificationdictionary | {"regression"} | A dictionary that contains all classes to be learned. This field is not necessary for a regression problem, enter "regression" here. |
| activationfunctionname | {"regression"} | Insert "regression" here to make it clear that you want to solve a regression problem. |
| errorfunctionname | {"MSE"} | The choice of error function determines whether you want to solve a regression problem or a classification problem. You should choose the mean squared error (input "MSE").  The cross entropy is intended for classification problems. |
| architecturearray | ndarray | Desired architecture of the neural network. Regression problems have only input and output neuron, no hidden layer. Therefore insert an array [0]. |
| regularizationlambda | float | Regularization is not possible for regression problems in our framework. Enter `regularizationlambda=0`, then no regularization is applied to the network. |


**Second step: train your data**

Call this function via the previously created object of the network class with the desired parameters.
`optimizer.gradientdescent(alpha, iterations, x, y)`
| Parameter | Input | |
| ------ | ------ | ------ |
| alpha | float | Desired alpha. |
| iterations| int | Desired number of iterations. |
| x | ndarray | Training data set: An array where each row represents a sample and contains all features of the training dataset entered into it (each feature=one column. |
| y | ndarray | An array with as many rows as samples, containing the correct label for each sample of the data (ground truth). |


At the end of the function trained weights and bias can be accessed too:

| Command | Type | |
| ------ | ------ | ------ |
| netz.thetasarray | ndarray | Returns the trained weights. |
| netz.biasarray | ndarray | Returns the bias. |

## Example

The repository has an executable example [jupyter notebook](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/lectures/machine-learning/student-material/ws20/Team-03/o2-machine-learning-framework-package/-/blob/master/example/framework_example_MNIST.ipynb) that demonstrates the usage of our framework with an example.



_______________________________________________________________________
[Way](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/lectures/machine-learning/student-material/ws20/Team-03/machine-learning-team-03) back to the main repository
