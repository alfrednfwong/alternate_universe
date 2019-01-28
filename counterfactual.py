import numpy as np
import pandas as pd
from helpers import timer



class Diamond():
    '''
    A diamond object in an expanding diamond search. Numerical discrete (int)
    features not yet implemented.
    '''

    def __init__(self, initial_vals, norm, df_features_attrs):
        self.initial_vals = initial_vals
        self.norm = norm

        # this part is replaced, but saved in case i'll need it later
        # self.bin_features = np.array(
        #     df_features_attrs[df_features_attrs.feature_type == 'binary'].index
        # )
        # self.mcat_features = np.array(
        #     df_features_attrs[df_features_attrs.feature_type == 'mcat'].index
        # )
        # self.cont_features = np.array(
        #     df_features_attrs[df_features_attrs.feature_type == 'cont'].index
        # )

        self.bin_features_vec = df_features_attrs.feature_type == 'binary'
        self.mcat_features_vec = df_features_attrs.feature_type == 'mcat'
        self.cont_features_vec = df_features_attrs.feature_type == 'cont'
        self.dist_wgts = df_features_attrs.distance_weight
        self.searchable_cat_features_vec = (
            (self.dist_wgts <= self.norm)
            & (self.bin_features_vec | self.mcat_features_vec)
        )
        self.num_cont_features = sum(self.cont_features_vec)

        self.u_bounds = df_features_attrs.u_bound
        self.l_bounds = df_features_attrs.l_bound
        self.num_cats = df_features_attrs.num_cats
        # create bounds in relative terms to initial_vals in distance weighted
        # terms
        self.rel_u_bounds_dist = (
            (self.u_bounds - self.initial_vals) * self.dist_wgts
        )
        self.rel_l_bounds_dist = (
            (self.l_bounds - self.initial_vals) * self.dist_wgts
        )
        # frequencies of changes of all the searchable categorical features
        #  that should appear in the search points
        # the sampling ratio for the searchable features , coded as
        # top / (top + bottom) here, is actually
        #######################################################################
        #      (c - 1)(r - w)**(d-1)
        # _________________________________
        # r**(d-1) + (c - 1)(r - w)**(d-1)
        #
        #
        # where c is num_cats (number of categories in that feature)
        # d is num_cont_features (number of continuous features in the model)
        # r is the radius / norm constraint
        # w is distance weights
        #######################################################################
        top = (
            (self.num_cats - 1)
            * (
                (self.norm - self.dist_wgts)
                ** (self.num_cont_features - 1)
            )
        )
        bottom = self.norm ** (self.num_cont_features - 1)
        self.sampling_ratios = np.array(
            self.searchable_cat_features_vec * (top / (top + bottom))
        )



    def __repr__(self):
        return (f'Diamond object in an expanding diamond search, with L1 norm '
                f'{self.norm}')

    # maybe unnecessary
    def dist_to_val(self, dist):
        '''
        Converts a search distance vector into actual feature values
        :param dist: array-like (num_features,)
        :return: array-like (num_features)
        '''
        return dist / self.dist_wgts

    def get_rand_cat_features(self):
        while True:
            result = ((
                np.random.rand(self.sampling_ratios.shape[0])
                < self.sampling_ratios
             ) * 1)
            # have to make sure we wont sample too many 1s that they sum up to
            # greater than the norm
            if result.sum() <= self.norm:
                return result

    def generate_point_candidate(self):
        '''
        Generate a search point that may or may not be out of feature bounds.
        :return:
        '''
        # the distance-weighted change vector with only the categorical
        # variables
        cand = self.get_rand_cat_features().astype('float64')
        norm_remaining = self.norm - cand.sum()
        # randomize the direction of change for the continuous variables
        change_signs = np.ones(self.num_cont_features)
        change_signs[(np.random.rand(self.num_cont_features) >= 0.5)] = -1
        # randomly split 1 into as many chunks as there are cont. features
        slices = np.sort(np.random.rand(self.num_cont_features - 1))
        cont_features_changes = np.append(slices, 1) - np.append(0, slices)
        # do the changes for the cont. features
        cand[self.cont_features_vec] = (
            norm_remaining * cont_features_changes * change_signs
        )
        return cand

    def generate_valid_search_point(self):
        # loop until we find a search point that is within all bounds
        while True:
            point = self.generate_point_candidate()
            if (
                (not (point > self.rel_u_bounds_dist).any())
                and (not (point < self.rel_l_bounds_dist).any())
            ):
                return point
