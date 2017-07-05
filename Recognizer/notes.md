# Artificial Intelligence Engineer Nanodegree - Probabilistic Models

## Introduction


## Features
### Ground (_to nose_)

### Cartesian Coordinates
### Polar Coordinates
### Delta Difference
### Custom
 - Feature Extraction: 
 	* some concept of the shape of the hand
 	* angle of the hand relative to horizontal considered necessary
 	
 - Feature vectors to consider:
 	* Each hand's x and y position
 	* angle of axis of least inertia
 	* eccentricity of bounding ellipse
## Model Selectors/Complexity Selection
Identify the "best" model or the "best" complexity.

complexity = flexibility to fit and explain data

### Constant
"dumb" selector but fast

### Cross Validation
Use Kfold cross validation to find the "best" model(# of hidden states) with the highest log likelihood

#### Pro/Con
- Pro
	* Uses all data to train
	* Avoid overfitting by having a validation set
	* Improves predictive accuracy over simpler selectors
	* Simple, general 
- Con
	* Slow to train compared to BIC
	* Doesn't add domain knowledge or iextra insight for inference -- purely concerened with performance and repeatability
	* Doesn't follow Occam's razor: may choose complex models with the highest log likelihood as opposed to less complex models with comparable log likelihoods

### Bayesian Information Criteria (BIC)
BIC approximates the Bayesian posterior probability of a candidate model.

A penalty term penalizes complexity to avoid overfitting, so that it is not necessary to also use cross-validation in the selection process

BIC follows the Occam's razor principle: a model should be simple enough for efficient computation and complex enough to capture data specifics.

The goal of BIC is to maximize the likelihood of the data while penalizing large-size models(complexity).

- BIC EQ = -2log L + plogN
> L : likelihood of the fitted model
> p: number of parameters
> N: number of data points

	* _-2logL_ decreases with increasing model complexity(more parameters), whereas the penalties _2p_ or _plogN_ increase with increasing complexity
	* BIC applies a large penalty when N > e^2 = 7.4

Model selection => the lower the BIC value the better the model.

#### Pro/Con
- Pro
	* Avoids overfitting by penalizing complex models
	* Faster than CV
	* Follows Occam's razor principle(bias towards simpler models)
- Con
	* Doesn't bring in information from competting models but indirectly factors complexity for each model
	* BIC applies a large penalty when N > e^2 = 7.4

### Deviance Information Criterion(DIC)
Occam's razor principle does not necessarily select the best performing model when used with classification. In particular, model selectors like BIC estimate using within-clas statistics without regard to competing classifiication categories.

The goal of DIC is not to select the simplest model that best explains the data(BIC) but to select the model that is less likely to have generated data belonging to competing classification categories. Complexity is defined as a measure of effective numbers of parameters.

The model selection criteria is **discriminant** because the model is selected in regard to the classification task by making use of the data that belong to the competing classes, thus introducing knowledge of the classification process in the model selection process.

Overall, we need to find the model(number of components) where the difference between competing models is the largest.

- DIC = log(P(original)) - avg(log(P(competing))

#### Pro/Con
- Pro
	* Adds more information to the selection process without adding more feature parameters(complexity)
	* Takes advantage of information from competing models(directly)
	* Biased towards simpler models

- Con
	* Slower than BIC
	* May overfit with complex models
	* If competiting models have similiar log likelihood, then the selection criterion doesn't add much more information. In this case, we would prefer BIC. However, if the difference between competitng models is low then our model isn't complex enough to represent the data points.T