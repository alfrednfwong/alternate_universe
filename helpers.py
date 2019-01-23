import numpy as np

class Feature:
    '''
    A feature in the model. But not necessarily a column in the data. A
    categorical feature with one-hot encoding may have multiple binary columns
    in the data, but here it is just one object.
    '''

    def __init__(
            self, indices, name, value, type='float', u_bound=None, l_bound=None
    ):
        self.indices = np.array(indices)
        self.name = name
        self.value = value
        self.type = type
        self.u_bound = u_bound
        self.l_bound = l_bound
        self.pos_side_done = False
        self.neg_side_done = False

    def __repr__(self):
        return (f'Feature object, index {self.index}')

    def perturb(x_data, step, pos_side):
        '''

        :param step:
        :param pos_side:
        :return:
        '''

        ### savegame. multiclass features are hard to handle. i can map the
        # object to change one hot encoding. But mapping from the data to the
        # object requires hard coding for every feature. and messy
        # maybe manually make a general feature map for every dataset, then
        # use it from both sides?
        # anyway, at this point i was going to leave behind this problem and
        # go with the bruteforce mock implementation, just to see the speed
        # i planned to fix the boolean feature problem here but maybe i can
        # skip that too? since it's just a speed test?
        # actually if i forget about all these possible problems, the speed test
        # should be done in a couple hours. prolly should do that first.





def perturb(x, feature_num, feature_value):
    '''
    Takes a vector of feature values x, changes one of the values and return the
    vector.
    :param x: array-like, vector of feature values
    :param feature_num: index of feature in x to be changed
    :param feature_value: new value of the feature.
    :return:
        numpy array - vector of feature values
    '''
    x[feature_num] = feature_value
    return x
