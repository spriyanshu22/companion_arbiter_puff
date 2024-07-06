import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
	
	feat = my_map(X_train)
	model = LogisticRegression(penalty='l2', C=120, fit_intercept=True, max_iter=2500)
	model.fit(feat, y_train)
	w = model.coef_[0]
	b = model.intercept_
	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
	return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	X = np.hstack((X, np.ones((X.shape[0], 1))))
	X = 1 - 2 * X

	cumprod_X = np.cumprod(X[:, ::-1], axis=1)[:, ::-1]

	feat = np.zeros((X.shape[0], X.shape[1] * (X.shape[1] - 1) // 2))

	index = 0

	for i in range(X.shape[1]):
		for j in range(i + 1, X.shape[1]):
			feat[:, index] = cumprod_X[:, i] * cumprod_X[:, j]
			index += 1

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	
	return feat