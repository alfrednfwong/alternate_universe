# Alternate Universe

A model-agnostic algorithm to provide counterfactual examples to a classified datapoint in a machine learning model, that serves as a human interpretable aid to understand what a supposedly black box model is doing.

e.g. A customer's loan application is rejected by the bank's decision algorithm and he invokes the GDPR's right to explanation clause and requests an explanation. Our system will generate combinations of features, which are as close to the applicant's as possible, and yet would get the decision algorithm to return the "approve" class, like "Had your income been 40k instead of 32k, we would have approved the loan."

From our research, systems that generate counterfactual example out there generally use one of three mechanisms:

Gradient based: set target variable to the alternative value, and compute the gradient of the cost function with respect to the input values, usually with an L1 norm constraint to encourage sparseness. This method is fast and straight forward but requires access to the model's gradient, and obviously requires the model to have a gradient. Also it cannot handle categorical variables.

Randomized search: Laugel et al(2017) proposed a "Growing Sphere" algorithm where search points in the input space are generated and fed into the inference system for an output until the alternative outcome is obtained. The search points are constrained within an n-ball with a certain radius, and if none of the search points are counterfactuals, the radius is increased and the process repeated. There is no mention of how categorical variables or possible bounds feature values can be handled.

Meta-feature comparison in decision trees: For the data point of interest, all leaves in the decision tree of the opposite class are listed and all the "meta-features" that lead to those leaves from the root are gathered and compared. The set of meta-features that are the closest (in L1 norm) to the data point of interest is selected. This requires the model to be tree based and access to the parameters are needed.

In this project we aim at providing model-agnostic counterfactual examples that can include continuous, binary and multicategorical variables. Our plan is to implement, from scratch, an algorithm we call the "expanding diamond", which is inspired by the Growing Sphere algorithm. Instead of sampling within an n-ball (L2 norm <= constant), we sample the surface of an "n-diamond" (L1 norm = constant), which should result in more cases closer to corners and edges that give sparse vectors. Also we plan to use a sample-and-reject mechanism to handle feature value bounds, and a boolean sampling algorithm to handle binary and multicategorical features.
