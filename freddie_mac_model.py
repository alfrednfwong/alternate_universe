import numpy as np
import pandas as pd
import pickle
from counterfactual import ClassifierModel
from copy import copy

PATH = './production/'
# if the predicted probability is >= this threshold, the classifier will return
# a 1, else 0
CF_THRESHOLD = 0.2

class FreddieMacModel(ClassifierModel):
    '''
    A subclass just to hold a specific model. The model is a classifier model
    that predicts the probability of a home mortgage loan being delinquent for
    two or more months at least once in it's life time. The data came from
    Freddie Mac, and the classifier is an ensemble of logistic regression and
    random forest, with soft voting.

    The class contains the classifier object,one-hot encoder, feature attributes
    and the method to process input features before feeding them into the
    classifier for prediction.
    '''

    def __init__(self):

        # it's all pickled
        # dict of lists of possible values for multicat features
        infile = open(PATH + 'mcat_cats.pkl', 'rb')
        self.mcat_cats = pickle.load(infile)
        infile.close()

        # feature attributes
        self.df_features_attrs = pd.read_csv(
            PATH + 'feature_attrs_freddie_mac.txt', sep='\t',
            index_col='feature_index'
        )

        # the one hot encoder
        infile = open(PATH + 'ohc.pkl', 'rb')
        self.ohc = pickle.load(infile)
        infile.close()

        # the classifier object
        infile = open(PATH + 'clf.pkl', 'rb')
        self.clf = pickle.load(infile)
        infile.close()

        self.mcat_columns = (list(
            self.df_features_attrs[
                self.df_features_attrs.feature_type == 'mcat'
            ].index
        ))

        self.decision_threshold = CF_THRESHOLD
        # dicts to translate to and from feature_index to feature name
        self.i_to_name = self.df_features_attrs['name'].to_dict()
        self.name_to_i = (
            self.df_features_attrs.reset_index().set_index('name')
            ['feature_index'].to_dict()
        )


    def __repr__(self):
        return f'Model to predict deliquency from the Freddie Mac data.'

    def process_inputs(self, arr_x):
        '''
        Encode dti into 2 variables, and do one hot encoding for the multicat
        variables
        :param arr_x: numpy array (num_raw_features,) datapoint to be processed
        :return: numpy array (num_input_vars,) array that can be fed into
           the model for training or prediction
        '''
        processed = copy(arr_x)
        # index 2 is dtl, or debt-to-loan ratio, in percentages.
        processed = np.hstack(
            (processed, ((processed[:, 2] > 65) * 1).reshape(-1, 1))
        )
        # in the original dataset, anything above 65 are recorded as not
        # applicable and encoded as 999
        processed[:, 2] = (np.apply_along_axis(
            (lambda x: (x != 999) * x), axis=0, arr=processed[:, 2]
        ))
        # one hot encoding for the multicategorical features
        processed = np.hstack((
            processed, self.ohc.transform(processed[:, self.mcat_columns])
        ))
        processed = np.delete(processed, [self.mcat_columns], axis=1)
        return processed
