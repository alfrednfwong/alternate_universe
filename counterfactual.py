import numpy as np
import pandas as pd

class Diamond():
    '''
    A diamond object in an expanding diamond search. Numerical discrete (int)
    features not yet implemented.
    '''
    def __init__(self, initial_vals, norm, df_features_attrs):
        self.initial_vals = initial_vals
        self.norm = norm
        self.bin_features = np.array(
            df_features_attrs[df_features_attrs.feature_type == 'binary'].index
        )
        self.multicat_features = np.array(
            df_features_attrs[df_features_attrs.feature_type == 'multicat'].index
        )
        self.cont_features = np.array(
            df_features_attrs[df_features_attrs.feature_type == 'cont'].index
        )
        self.distance_weights = df_features_attrs.distance_weight
        self.u_bounds = df_features_attrs.u_bound
        self.l_bounds = df_features_attrs.l_bound
        self.m_cat_values = df_features_attrs.multicat_possible_values


    def __repr__(self):
        return (f'Diamond object in an expanding diamond search, with L1 norm '
                f'{self.norm}')

    def generate_point(self):
        '''
        Generate a search point that may or may not be out of feature bounds.
        :return:
        '''
        proposed_search_point = self.initial_vals.copy()
        ### savegameXXX now do "list all binary / cat vars , unlist if distance>r"


        return