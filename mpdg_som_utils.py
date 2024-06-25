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

def update_weight_vectors(weight_vectors,
                          learning_rate_function,
                          neighborhood_function,
                          data_vectors,
                          index,
                          step):

    current_weight_vectors = weight_vectors.copy()
    updated_weight_vectors = current_weight_vectors +\
                             learning_rate_function(step) * neighborhood_function(step, index) *\
                            (data_vectors - current_weight_vectors)

    return updated_weight_vectors

class SOM_LearningRateFunctions:

    def power_law_lrf(step,
                  maximum_steps, learning_rate):
        
        return learning_rate ** (step/maximum_steps)
