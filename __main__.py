from freddie_mac_model import FreddieMacModel
import counterfactual as cf
import numpy as np

def main(vals, params, to_freeze=[]):
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
    cfq.model.df_features_attrs.loc[to_freeze, 'feature_type'] = 'frozen'

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
        features_changed, initial_proba, initial_decision, cf_example, cf_proba,
        cf_decision, found_norm, num_found
    )


if __name__ == '__main__':
    main()