import numpy as np
import pandas as pd
import pickle
from counterfactual import ClassifierModel
from copy import copy

PATH = './production/'
CF_THRESHOLD = 0.2

class FreddieMacModel(ClassifierModel):

    def __init__(self):

        infile = open(PATH + 'mcat_cats.pkl', 'rb')
        self.mcat_cats = pickle.load(infile)
        infile.close()

        self.df_features_attrs = pd.read_csv(
            PATH + 'feature_attrs_freddie_mac.txt', sep='\t',
            index_col='feature_index'
        )

        infile = open(PATH + 'ohc.pkl', 'rb')
        self.ohc = pickle.load(infile)
        infile.close()

        infile = open(PATH + 'clf.pkl', 'rb')
        self.clf = pickle.load(infile)
        infile.close()

        self.mcat_columns = (list(
            self.df_features_attrs[
                self.df_features_attrs.feature_type == 'mcat'
            ].index
        ))

        self.decision_threshold = CF_THRESHOLD
        self.name_to_i = (
            self.df_features_attrs.reset_index().set_index('name')
            ['feature_index'].to_dict()
        )
        self.i_to_name = self.df_features_attrs['name'].to_dict()


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
        processed = np.hstack(
            (processed, ((processed[:, 2] > 65) * 1).reshape(-1, 1))
        )
        processed[:, 2] = (np.apply_along_axis(
            (lambda x: (x != 999) * x), axis=0, arr=processed[:, 2]
        ))
        processed = np.hstack((
            processed, self.ohc.transform(processed[:, self.mcat_columns])
        ))
        processed = np.delete(processed, [self.mcat_columns], axis=1)
        return processed
