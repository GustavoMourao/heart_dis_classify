import split_set_data
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def metric_evaluation_model(classifier, yTest, xTest):
    '''
    Evaluation the models from the ROC curve (adopted evaluation metric).
    '''    

    # Load pretrained model and return predicted values.
    loaded_model = joblib.load(classifier)
    predicted = loaded_model.predict_proba(xTest)

    # Calculates ROC curve.
    fpr, tpr, _ = roc_curve(yTest[:,:], predicted[:,1])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve.
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def main():
    '''
    Main function
    '''

    # Get split data.
    yTrain, xTrain, yTest, xTest = split_set_data.get_raw_data()

    # Represents the metric evaluation (ROC curve) of each model.
    metric_evaluation_model('SVM_C_1e+16_gamma_1e-11.pkl', yTest, xTest)
    metric_evaluation_model('ANN_nlayer_1190.pkl', yTest, xTest)

if __name__=='__main__':

    '''
    Main function
    '''
    main() 