# heart_dis_classify

heart_dis_classify is a project that aims to that classify heart disease based on 13 attributes. 

The set of data is provided courtesy of the Cleveland Heart Disease Database via the UCI Machine Learning repository [1].

### This project uses this dependences:

* [python] : 3.7.1 
* [scikit-learn] : 0.20.0
* [matplotlib] : 3.2.0
* [numpy] : 1.15.4
* [pytest] : 4.5.0

User instalation:
```sh
pip install scikit-learn
```
and so on for the other packages..

### Model description

It was applied/compared two models:

  - ANN (MLP): Artificial Neural Network (Multilayer Perceptron)
  - SVM: Support Vector Machine 

#### Overview of Models 
The project evaluates the best parameters for each model. 

In this direction, from de SVM model it is evaluate the influence of parameters C (soft margin cost function) and gamma (Kernel coefficient) on accuracy of training dataset.

From other side the ANN mode it is evaluate through 3 hidden layers. In this sense, is evaluated the influence of the number of neurons on first layer (by keep 500 neurons on the other two layers). The best models is saved based on a precision threshold of  85% [2].

### Overview of Files

First Tab:
> eval_ANN.py: Obtain (train and save) best ANN models from comparing the highest precision.[2]

> eval_SVM.py: Obtain representation of best parameters (C and gamma) of SVM that returns high accuracy. Save model with best features.

> split_set_data.py: Separates raw data between train (70%) and test(30%) data. It wasn't used dev data in this version.

> split_set_data.py: Separates raw data between train (70%) and test(30%) data. It wasn't used dev data in this version.

> metric_eval_response_ROC.py: Evaluation the models from the ROC curve (adopted evaluation metric). ROC (Receiver Operating Characteristics) is a widely metric used to evaluate the performance of binary classificators. As higher the Area Under The Curve (AUC), better is the model.

> unit_test_eval_methods.py: Unit test that evaluates the SVM and ANN models. It's expected that, the average precision of the saved models (as .pkl file) returns a precision higher than 60% (for SVM) and higher than 80% (for ANN).


#### Partial Results
> SVM paramaters analysis:

![Image1](https://github.com/GustavoMourao/heart_dis_classify/blob/master//results_graphs/SVM_parameters.png)

> ROC curve ANN:

![Image2](https://github.com/GustavoMourao/heart_dis_classify/blob/master/results_graphs/SVM_ROC.png)


> ROC curve ANN:

![Image3](https://github.com/GustavoMourao/heart_dis_classify/blob/master/results_graphs/ANN_ROC.png)


#### Sources
>[1] Aha, D., and Dennis Kibler. "Instance-based prediction of heart-disease presence with the Cleveland database." University of California 3.1 (1988): 3-2.

>[2] Graber, Mark L. "The incidence of diagnostic error in medicine." BMJ Qual Saf 22.Suppl 2 (2013): ii21-ii27.

