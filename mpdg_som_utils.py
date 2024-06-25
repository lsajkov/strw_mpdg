import numpy as np
from numpy import linalg as linalg


def chi_sq_dist(weight_vector,
                data_vector,
                covar_vector = None,
                use_covariance = False,
                data_dim = None,):
    
    if (~use_covariance) & (data_dim != None):
        covar_matrix = np.diagflat([1] * data_dim)
    
    elif (use_covariance) & (covar_vector is None):
        raise(ValueError('There is no covariance matrix for the given data vector!'))

    covar_matrix = np.diagflat(covar_vector)
    inv_covar_matrix = linalg.inv(covar_matrix)

    vector_difference = data_vector - weight_vector

    return np.dot(np.dot(vector_difference, inv_covar_matrix),vector_difference)

def update_weights(
        
                    ):
    
    