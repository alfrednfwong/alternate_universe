from freddie_mac_model import FreddieMacModel
import counterfactual as cf
import numpy as np

def main(vals, params, to_freeze=[]):
    '''
    Main function for the deployment of counterfactual example search on the
    Freddie Mac mortgage delinquency classifier model.
    :param vals: dict. Feature values of the data point of interest.
    :param params: dict. Search parameters.
    :param to_freeze: list. List of feature names to freeze in the cf search
    :return: tuple of:
      float. probability of the initial data point being of class 1.
      int. predicted class of the initial data point
      dict. names of features changed in the cf example as keys, the new values
        as values.
      numpy array, [num_features] the values of the counterfactual example
      float. probability of the cf example being of class 1
      int. predicted class of the cf example
      float. the norm at which the cf example is found
      int. the number of (unsparsified, not all returned) cf examples found at
        that norm.
    '''
    model = FreddieMacModel()
    initial_vals = np.array([
        vals['bought_home_before'],
        vals['credit_score'],
        vals['dti'],
        vals['is_insured'],
        vals['ltv'],
        vals['multi_borrowers'],
        vals['orig_rate'],
        vals['prop_type'],
        vals['purpose'],
        vals['state']
    ], dtype=object)
    cfq = cf.CfQuestion(
        initial_vals=initial_vals,
        ppd=params['ppd'],
        start_norm=params['start_norm'],
        step_size=params['step_size'],
        max_iter=params['max_iter'],
        sparsifier_buffer=params['sparsifier_buffer'],
        model=model
    )
    # to_freeze comes in feature names. Here we make list of indices out of it.
    i_to_freeze = []
    for elem in to_freeze:
        i_to_freeze.append(model.name_to_i[elem])
    cfq.model.df_features_attrs.loc[i_to_freeze, 'feature_type'] = 'frozen'

    cf_result = cfq.solve()
    if cf_result:
        cf_example, cf_proba, found_norm, num_found = cf_result
    features_changed = {}
    for i in np.where(cf_example != initial_vals)[0]:
        features_changed[model.i_to_name[i]] = cf_example[i]

    initial_proba = cfq.predict_point(initial_vals, 1)
    initial_decision = (initial_proba > model.decision_threshold) * 1
    cf_decision = (cf_proba > model.decision_threshold) * 1
    print(f'Sparsified point is\n{cf_example}\n')
    print(f'with a predicted probability of {cf_proba}\n')
    print(f'Features changed:\n{features_changed}')
    return (
        initial_proba, initial_decision, features_changed, cf_example, cf_proba,
        cf_decision, found_norm, num_found
    )


if __name__ == '__main__':
    main()