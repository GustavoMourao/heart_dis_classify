import split_set_data
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score
import unittest

class MethodsPrecisionTest(unittest.TestCase):
    '''
    Class that implements two methods of evaluation SVM and ANN models.
    SVM: Precision has to higher than 60%.
    ANN: Precision has to higher than 80%.
    '''

    def test_SVM_output(self):
        '''
        SVM test eval.
        '''

        # Get segmented data.
        yTrain, xTrain, yTest, xTest = split_set_data.get_raw_data()

        # Load pretrained model and return predicted values.
        loaded_model = joblib.load('SVM_C_1e+16_gamma_1e-11.pkl')
        predicted = loaded_model.predict_proba(xTest)

        # Calculates precision.
        avrPrec = average_precision_score(yTest[:,:], predicted[:,1])    

        # Verify if model returns ate least 60% of average precision.
        assert avrPrec > 60/100

    def test_ANN_output(self):
        '''
        ANN test eval.
        '''

        # Get segmented data.
        yTrain, xTrain, yTest, xTest = split_set_data.get_raw_data()

        # Load pretrained model and return predicted values.
        loaded_model = joblib.load('ANN_nlayer_1190.pkl')
        predicted = loaded_model.predict_proba(xTest)

        # Calculates precision.
        avrPrec = average_precision_score(yTest[:,:], predicted[:,1])    

        # Verify if model returns ate least 80% of average precision.
        assert avrPrec > 80/100
    
if __name__=='__main__':
    '''
    Main function
    '''
    unittest.main() 