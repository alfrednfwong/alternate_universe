import numpy as np
import pandas as pd
from copy import copy
from helpers import timer


class Diamond():
    '''
    A diamond object in an expanding diamond search. Numerical discrete (int)
    features not yet implemented.
    '''

    def __init__(self, initial_vals, norm, df_features_attrs, mcat_alt_cats):
        self.initial_vals = initial_vals
        self.norm = norm
        self.is_bin = np.array(df_features_attrs.feature_type == 'binary')
        self.is_mcat = df_features_attrs.feature_type == 'mcat'
        self.is_cont = np.array(df_features_attrs.feature_type == 'cont')
        self.dist_wgts = df_features_attrs.distance_weight
        # array of booleans to indicate which categorical features can be
        # changed, ie, changing any one of them wont exceed the norm for this
        # diamond
        self.is_searchable = (
            (self.dist_wgts <= self.norm)
            & (self.is_bin | self.is_mcat)
        )
        self.num_cont_features = sum(self.is_cont)
        self.num_cats = df_features_attrs.num_cats
        # dict of lists of all possible values of mcat features, except those
        # appearing in initial_vals
        self.mcat_alt_cats = mcat_alt_cats
        # create bounds in relative terms to initial_vals in distance weighted
        # terms
        self.rel_u_bounds_dist = (
            (df_features_attrs.u_bound - self.initial_vals) * self.dist_wgts
        )
        self.rel_l_bounds_dist = (
            (df_features_attrs.l_bound - self.initial_vals) * self.dist_wgts
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
            self.is_searchable * (top / (top + bottom))
        )

    def __repr__(self):
        return (f'Diamond object in an expanding diamond search, with L1 norm '
                f'{self.norm}')

    def get_rand_cat_features(self):
        '''
        Generate random change/not change values for the categorical features.
        Cont.values will always get a 0
        :return: np.array (num_features,)
        '''
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
        :return: np.array (num_features,)
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
        cand[self.is_cont] = (
            norm_remaining * cont_features_changes * change_signs
        )
        return cand

    def generate_search_point(self):
        '''
        Calls the generate_point_candidate(), get a point candidate, evaluate
        if it is within feature value bounds. If it is, encode it into a search
        point that can be fed into the inference pipeline; else get another
        candidate.
        :return: pd.Series(num_features,)
        '''
        # loop until we find a search point that is within all bounds
        # then encode the changes to a format readable by the ML inference
        # pipeline
        while True:
            # delta records the distance to change for cont features, and
            # whether or not (1 or 0) for the bin and mcat features
            # convert into series because we need the index for tracing feature
            # number for the mcat features
            deltas = pd.Series(self.generate_point_candidate())
            point = pd.Series(copy(self.initial_vals))
            if (
                (not (deltas > self.rel_u_bounds_dist).any())
                and (not (deltas < self.rel_l_bounds_dist).any())
            ):
                # mcat features. for all the mcat features, delta contains 1s
                # and 0s. 1 for change and 0 for no change, first we record
                # which mcat features to change
                mcats_to_change = (deltas == 1) & self.is_mcat
                # then change the cats
                for i in mcats_to_change[mcats_to_change].index:
                    # pick one random from the mcat_alt_cats lists, which
                    # contain all cats except the ones in the initial values
                    point[i] = np.random.choice(self.mcat_alt_cats[i])
                point[self.is_bin] = (
                    # have the binary rows of both deltas and initial values in
                    # booleans. XOR them and we'll get the right outcome, then
                    # * 1 to get back an integer
                    deltas[self.is_bin].astype('bool')
                    ^ self.initial_vals[self.is_bin]
                ) * 1
                point[self.is_cont] = (
                    # (delta / weight) + initial_value = new_value
                    (deltas[self.is_cont] / self.dist_wgts[self.is_cont])
                    + self.initial_vals[self.is_cont]
                )
                return point
