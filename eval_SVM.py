from sklearn.externals import joblib
import split_set_data
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_SVM_models(yTrain, xTrain, yTest, xTest):
    '''
    Evaluates the best parameter gamma and C on SVM model.
    '''
    c_range = np.logspace(-2, 10, 18)
    gamma_range = np.logspace(-12, 3, 13)
    param_grid = dict(gamma=gamma_range, C=c_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(xTrain, yTrain)   

    c_range_eval = [1e-2, 1, 1e2]
    gamma_range_eval = [1e-1, 1, 1e1]
    classifiers = []
    for C in c_range_eval:
        for gamma in gamma_range_eval:
            clf = svm.SVC(C=C, gamma=gamma)
            clf.fit(xTrain, yTrain)
            classifiers.append((C, gamma, clf))     

    scores = grid.cv_results_['mean_test_score'].reshape(len(c_range),
                                                     len(gamma_range))

    # Plot parameters and score matrix.
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.get_cmap('Blues', 6))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(c_range)), c_range)
    plt.title('Validation accuracy')
    plt.show()
    #plt.savefig('SVM_evaluation_parameters.png')

    # Save model with best paramater
    classifier = svm.SVC(C=1e16, gamma=1e-11, kernel='rbf', probability=True)
    classifier.fit(xTrain, yTrain)
    joblib.dump(classifier, 'SVM_C_'+ str(1e16) + '_gamma_' + str(1e-11) +'.pkl')

if __name__=='__main__':
    '''
    Main function
    '''
    # Get split data.
    yTrain, xTrain, yTest, xTest = split_set_data.get_raw_data()

    ## Preprocessing data: FAIL!
    #xTrain = preprocessing.scale(xTrain)
    #xTest = preprocessing.scale(xTest)

    # Save best ANN models. 
    train_SVM_models(yTrain, xTrain, yTest, xTest)
