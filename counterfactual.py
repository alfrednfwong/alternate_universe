import numpy as np
import pandas as pd
from copy import copy, deepcopy
from helpers import timer


class Diamond():
    '''
    A diamond object in an expanding diamond search for counterfactual examples.
    Numerical discrete (int) features not yet implemented.

    It takes in the initial feature values, a norm value, attributes and
    (optional) value bounds of the features, then generate a random point that
    has an L1 norm equal to the norm value from the initial point. The points
    sampled are evenly distributed over the surface of the "n-diamond"
    '''

    def __init__(self, initial_vals, norm, df_features_attrs, mcat_alt_cats):
        self.initial_vals = initial_vals
        self.norm = norm
        # note that in df_features_attrs, some feeature_type values may be
        # changed to 'frozen', as inputted by the user, to prevent the value
        # to be changed along that dimension. Otherwise these 3 types are
        # exhaustive
        self.is_bin = np.array(df_features_attrs.feature_type == 'binary')
        self.is_mcat = df_features_attrs.feature_type == 'mcat'
        self.is_cont = np.array(df_features_attrs.feature_type == 'cont')
        # weights to indicate the significance of changing each feature by 1.
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
            # have to make sure we wont sample too many ones that they sum up to
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
        # now that the mcat and binary variables are determined, to make sure
        # the norm of the resulting point is equal to the diamond's norm, we
        # divide norm - sum_of_weighted_changes_in_cat_features evenly among
        # the cont features and make them change by that much, weighted.
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
            # check to reject points that are out of bounds
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
                # have the binary rows of both deltas and initial values in
                # booleans. XOR them and we'll get the right outcome, then
                # * 1 to get back an integer
                point[self.is_bin] = (
                    (deltas[self.is_bin].astype('bool') ^
                     self.initial_vals[self.is_bin]) * 1
                )
                # (delta / weight) + initial_value = new_value
                point[self.is_cont] = (
                    (deltas[self.is_cont] / self.dist_wgts[self.is_cont])
                    + self.initial_vals[self.is_cont]
                )
                return point, deltas


class Sparsifier():
    '''
    Takes a counterfactual example and tries to sparsify it, ie, to revert as
    many features as possible to the initial value while still being a
    counterfactual example.
    '''

    def __init__(self, dense_point, initial_vals, cf_y, cf_threshold, changed):
        # the counterfactual example before sparsification
        self.dense = copy(dense_point)
        # the datapoint of interest, for which the counterfactuals are required
        self.initial_vals = initial_vals
        # the desired (counterfactual) output value
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

    def what_if_reduce(self, predict):
        '''
        Returns how "trivial" are the change of each of the features in
        contributing to getting a counterfactual outcome, ie, how unimportant
        they are.
        A positive value means if we change the value of that feature in the
        current point back to the initial value, the outcome will still be
        counterfactual. The larger the triviality value, the more confident
        is the model in predicting the counterfactual.
        A negative value means reducing that feature will cause the outcome to
        change from being a counterfactual to being the same as the initial
        prediction.
        :param predict: callable. A function to return the probability of a
            datapoint being of a specific class.
        :return: numpy array (num_features,)
        '''
        # initialize the array. for features that are not changed, we set as 0s
        # so that they'll be ignored
        trivialities = np.zeros_like(self.dense)
        # loop thru all changed features
        for i in self.changed:
            temp = copy(self.current)
            # set feature i to the initial value, cancelling the change along
            # that dimension
            temp[i] = self.initial_vals[i]
            # get the predicted probability of the desired class given the
            # cancellation, minus the counterfactual threshold (only when proba>
            # threshold will the data point be classified as the desired class)
            trivialities[i] = (
                predict(temp, self.cf_y) - self.cf_threshold
            )
        return trivialities

    def sparsify(self, predict):
        '''
        Changes as many features in the dense_point as possible back to the
        initial values, with the constraint that the output is still a
        counterfactual example. Returns the final sparsified point.
        :param predict: Callable. A function to return the probability of a
            datapoint being of a specific class. To be fed into
            self.what_if_reduce()
        :param clf. Classifier object
        :param ohc. One-hot encoder object
        :return: numpy array, (num_features,)
        '''
        while True:
            trivialities = self.what_if_reduce(predict)
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
                # remove the index of the reverted feature from the changed list
                self.changed = self.changed[self.changed != idx_to_reduce]


class CfQuestion():
    '''
    An object to wrap a single counterfactual example request. Takes in the
    feature values of the data point of interest, search parameters and a Model
    object, and returns a sparsified counterfactual example.
    '''

    def __init__(
        self, initial_vals, ppd, start_norm, step_size, max_iter,
        sparsifier_buffer, model
    ):
        # the data point for which the counterfactual example is requrested
        self.initial_vals = initial_vals
        # dict of lists of all possible values of mcat features, except those
        # appearing in initial_vals
        self.mcat_alt_cats = deepcopy(model.mcat_cats)
        for key in self.mcat_alt_cats.keys():
            self.mcat_alt_cats[key].remove(initial_vals[key])
        # points per diamond (norm value) to search
        self.ppd = ppd
        # norm of the first diamond (iteration)
        self.start_norm = start_norm
        # amount of norm to increase between diamonds
        self.step_size = step_size
        # if after this many diamonds and still no cf found, stop.
        self.max_iter = max_iter
        self.end_norm = start_norm + (step_size * (max_iter - 1))
        # model object that provides the classifier, one-hot encoder, input
        # processing function, etc.
        self.model = deepcopy(model)

        initial_proba = self.predict_point(initial_vals, 1)
        # target value / class of output to look for, which is the opposite of
        # the class of the initial point.
        self.cf_y = (initial_proba < self.model.decision_threshold) * 1
        # decision_threshold is used to compare with the probability of the data
        #  point being a 1.
        # To support searching for both classes 1 and 0, we have another
        # variable, cf_threshold, to represent the threshold for comparison
        # against is the probability of the data point being the target
        # class.
        # search_threshold is defined similarly to cf_threshold.
        if self.cf_y == 1:
            self.cf_threshold = self.model.decision_threshold
        else:
            self.cf_threshold = 1 - self.model.decision_threshold
        self.search_threshold = self.cf_threshold + sparsifier_buffer

    def __repr__(self):
        return f'Counterfactual question with for {self.initial_vals}'

    def predict_point(self, point, target):
        '''
        Returns the probability prediction for a single data point
        :param point, numpy array [num_features,] The data point to predict
        :param target, int. The target class.
        '''
        # stack to make it a 2d array, just to get around the problem that
        # numpy's vectorized operations behave differently on 1d arrays
        temp = np.vstack((point, point))
        probas = self.model.clf.predict_proba(
            self.model.process_inputs(temp)
        )
        # the 0 is there because proba will contain 2 sets of identical
        # probabilities
        return probas[0][target]

    def get_dense_cf(self):
        '''
        Repeatedly call the Diamond class until a counterfactual point is found.
        :return: tuple of:
          np.array [num_features,] counterfactual example, unsparsified
          list. indices of features changed
          float. the norm at which the cf point is found
          int. the number of cf points found in the same diamond. Used to
            indicate whether start_norm is set too high, step_size too high or
            ppd too high
        or Nonetype (when no cf point is found)
        '''

        for norm in np.linspace(self.start_norm, self.end_norm, self.max_iter):
            dia = Diamond(
                self.initial_vals, norm, self.model.df_features_attrs,
                self.mcat_alt_cats
            )
            print(f'Searching at norm {norm:.6}')
            # initialize the arrays that collect all the search points.
            # using initial_vals here just to make sure the dimensions are
            # compatible, and also that it is a point that will certainly
            # not be a countefactual.
            points = copy(self.initial_vals)
            # same idea here. delta with all 0s wont be a counterfactual.
            deltas = np.zeros_like(self.initial_vals)
            for _ in range(self.ppd):
                new_point, new_delta = dia.generate_search_point()
                points = np.vstack((points, new_point))
                deltas = np.vstack((deltas, new_delta))
            pred = self.model.clf.predict_proba(
                self.model.process_inputs(points)
            )
            # a boolean vector to indicate which of the search points are
            # classified as the desired counterfactual class
            are_cf = pred[:, self.cf_y] > self.search_threshold
            num_found = are_cf.sum()
            if are_cf.any():
                # cfes for Counter Factual ExampleS
                dense_cfes = points[are_cf]
                cfe_deltas = deltas[are_cf]
                # locate the sparsest (least features changed) of all the cfes
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
                return (dense_cfes[best_i], changed, norm, num_found)
        print(f'No counterfactual example found at norm {norm}')
        return None

    def solve(self):
        '''
        Main method of the CfQuestion object. Returns answers
        :return: tuple of:
          np.array [num_features,]. Sparsified cf example
          float. Probability the cf example is of class 1
          float. The norm at which the cf example is found
          int. the number of cf points found in the same diamond. Used to
            indicate whether start_norm is set too high, step_size too high or
            ppd too high
        or Nonetype (when no cf point is found)
        '''
        dense_result = self.get_dense_cf()
        if dense_result:
            dense, changed, found_norm, num_found = dense_result
        else:
            return None
        spr = Sparsifier(
            dense, self.initial_vals, self.cf_y, self.cf_threshold,
            changed
        )
        sparse = spr.sparsify(self.predict_point)
        # note that this proba is the probability of sparse being 1, not of it
        # being the desired class
        sparse_proba = self.predict_point(sparse, 1)
        return (sparse, sparse_proba, found_norm, num_found)


class ClassifierModel():
    '''
    Blank unused superclass for future use. Now the only subclass is one that
    holds a single model
    '''

    def __init__(self):
        pass

    def __repr__(self):
        pass
