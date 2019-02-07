import numpy as np
import pandas as pd
from copy import copy, deepcopy
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

        #######################################################################
        # frequencies of changes of all the searchable categorical features
        #  that should appear in the search points
        # the sampling ratio for the searchable features , coded as
        # top / (top + bottom) here, is actually
        #######################################################################
        #      (c - 1)(r - w)**(d-1)
        # _________________________________
        # r**(d-1) + (c - 1)(r - w)**(d-1)
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
        # replace nans with 0s
        self.sampling_ratios = np.nan_to_num(self.sampling_ratios)

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
            if (result * self.dist_wgts).sum() <= self.norm:
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
            point = copy(self.initial_vals)
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
                                         # have the binary rows of both
                                         # deltas and initial values in
                                         # booleans. XOR them and we'll get
                                         # the right outcome, then
                                         # * 1 to get back an integer
                                         deltas[self.is_bin].astype('bool')
                                         ^ self.initial_vals[self.is_bin]
                                     ) * 1
                point[self.is_cont] = (
                    # (delta / weight) + initial_value = new_value
                    (deltas[self.is_cont] / self.dist_wgts[self.is_cont])
                    + self.initial_vals[self.is_cont]
                )
                return point, deltas


class Sparsifier():

    def __init__(self, dense_point, initial_vals, cf_y, cf_threshold, changed):
        self.dense = copy(dense_point)
        self.initial_vals = initial_vals
        # the output value desired (counterfactual)
        self.cf_y = cf_y
        # the predicted probability threshold, above which the data point will
        # be classified as the desired class (ie, cf_y)
        self.cf_threshold = cf_threshold
        # array of indices of features changed
        self.changed = changed
        self.current = copy(dense_point)

    def __repr__(self):
        return (
            f'Counterfactual example object. With the dense (starting) point '
            f'{self.dense}'
        )

    def what_if_reduce(self, predict, clf, ohc):
        '''
        Returns how trivial are the change of each of the features in
        contributing to getting a counterfactual outcome, ie, how unimportant
        they are.
        A positive value means if we change the value of that feature in the
        current point back to the initial value, the outcome will still be
        counterfactual. The larger the triviality value, the more confident
        is the model in predicting the counterfactual.
        A negative value means reducing that feature will cause the outcome to
        change from being a counterfactual to being the same as the initial
        prediction.
        :return: numpy array (num_features,)
        '''
        # initialize the array. for features that are not changed, we set as 0s
        # so that they'll be ignored
        trivialities = np.zeros_like(self.dense)
        # loop thru all changed features
        for i in self.changed:
            temp = copy(self.current)
            # set feature i to the initial value, cancelling the change
            temp[i] = self.initial_vals[i]
            # get the predicted probability of the desired class given the
            # cancellation, minus the counterfactual threshold (only when proba>
            # threshold will the data point be classified as the desired class)
            trivialities[i] = (
                predict(temp, self.cf_y) - self.cf_threshold
            )
        return trivialities

    def sparsify(self, pred_, clf, ohc):
        '''
        Changes as many features in the dense_point as possible back to the
        initial values, with the constraint that the output is still a
        counterfactual example. Returns the final sparsified point.
        :return: numpy array, (num_features,)
        '''
        while True:
            trivialities = self.what_if_reduce(pred_, clf, ohc)
            # if none of the triviality ratings is positive, that means none
            # of the features can be sparsified
            if (trivialities <= 0).all():
                return self.current
            else:
                idx_to_reduce = trivialities.argmax()
                # revert the least important feature back to the initial value
                self.current[idx_to_reduce] = (
                    self.initial_vals[idx_to_reduce]
                )
                self.changed = self.changed[self.changed != idx_to_reduce]


class CfQuestion():

    def __init__(
        self, initial_vals, ppd, start_norm, step_size, max_iter,
        sparsifier_buffer, model
    ):
        self.initial_vals = initial_vals
        self.mcat_alt_cats = deepcopy(model.mcat_cats)
        for key in self.mcat_alt_cats.keys():
            self.mcat_alt_cats[key].remove(initial_vals[key])
        self.ppd = ppd
        self.start_norm = start_norm
        self.step_size = step_size
        self.max_iter = max_iter
        self.end_norm = start_norm + (step_size * (max_iter - 1))
        self.model = deepcopy(model)

        initial_proba = self.predict_point(initial_vals, 1)
        self.cf_y = (initial_proba < self.model.decision_threshold) * 1
        if self.cf_y == 1:
            self.cf_threshold = self.model.decision_threshold
        else:
            self.cf_threshold = 1 - self.model.decision_threshold
        self.search_threshold = self.cf_threshold + sparsifier_buffer

    def __repr__(self):
        return f'Counterfactual question with for {self.initial_vals}'

    def predict_point(self, point, target):
        '''Returns the probability prediction for a single data point'''
        # stack to make it a 2d array, just to get around the problem that
        # numpy's vectorized operations behave differently on 1d arrays
        temp = np.vstack((point, point))
        probas = self.model.clf.predict_proba(
            self.model.process_inputs(temp)
        )
        return probas[0][target]

    def get_dense_cf(self):

        for norm in np.linspace(self.start_norm, self.end_norm, self.max_iter):
            dia = Diamond(
                self.initial_vals, norm, self.model.df_features_attrs,
                self.mcat_alt_cats
            )
            print(f'Searching at norm {norm:.6}')
            points = copy(self.initial_vals)
            deltas = np.zeros_like(self.initial_vals)
            for _ in range(self.ppd):
                new_point, new_delta = dia.generate_search_point()
                points = np.vstack((points, new_point))
                deltas = np.vstack((deltas, new_delta))
            pred = self.model.clf.predict_proba(
                self.model.process_inputs(points)
            )
            are_cf = pred[:, self.cf_y] > self.search_threshold
            num_found = are_cf.sum()
            if are_cf.any():
                dense_cfes = points[are_cf]
                cfe_deltas = deltas[are_cf]
                best_i = np.argmin(np.count_nonzero(deltas, axis=1))
                # np.where returns a tuple for the x and y indices, thus the [0]
                changed = np.where(abs(cfe_deltas[best_i]) > 0)[0]

                print(
                    f'{num_found} counterfactual example(s) found at norm '
                    f'{norm:.6}\n'
                )
                print(
                    f'Best unsparsified point is\n{dense_cfes[best_i]}\n'
                )
                print(
                    f'with a predicted probability of '
                    f'{self.predict_point(dense_cfes[best_i], 1)}\n'
                )
                return (
                    dense_cfes[best_i], cfe_deltas[best_i], changed, norm,
                    num_found
                )
        print(f'No counterfactual example found at norm {norm}')
        return None

    def solve(self):
        dense_result = self.get_dense_cf()
        if dense_result:
            dense, _, changed, found_norm, num_found = dense_result
        else:
            return None
        spr = Sparsifier(
            dense, self.initial_vals, self.cf_y, self.cf_threshold,
            changed
        )
        sparse = spr.sparsify(
            self.predict_point, self.model.clf, self.model.ohc
        )
        sparse_proba = self.predict_point(sparse, 1)
        return sparse, sparse_proba, found_norm, num_found


class ClassifierModel():

    def __init__(self):
        pass

    def __repr__(self):
        '''Blank superclass'''
