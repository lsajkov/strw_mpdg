import numpy as np
from numpy import linalg as linalg


def chi_sq_dist(weight_vector,
                data_vector,
                covar_vector = None,
                use_covariance = False,
                data_dim = None,):
    
    if (~use_covariance) & (data_dim is not None):
        covar_matrix = np.diagflat([1] * data_dim)

    elif (~use_covariance) & (data_dim is None):
        raise(ValueError('Please pass a dimensionality for the data (number of variables).'))
    
    elif (use_covariance) & (covar_vector is None):
        raise(ValueError('There is no covariance matrix for the given data vector!'))

    elif (use_covariance) & (covar_vector is not None):
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

# def distance_calculator(mapsize,
#                         weight_vectors,
#                         data_vector,
#                         distance_metric,
#                         *args):

#     distance_map = np.full(mapsize, np.nan)
#     iteration_map = np.nditer(distance_map, flags = ['multi_index'])
    
#     for _ in iteration_map:
#         distance = distance_metric(weight_vectors[*iteration_map.multi_index],
#                                    data_vector,
#                                    *args)
#         distance_map[*iteration_map.multi_index] = distance

def find_closest_vector()

def training_step(weight_vectors,
                  ):

    current_weight_vectors = weight_vectors.copy()

    #
    #

    updated_weight_vectors = update_weight_vectors()
    
    return updated_weight_vectors

class SOM_LearningRateFunctions:

    def power_law_lrf(step,
                  maximum_steps, learning_rate):
        
        return learning_rate ** (step/maximum_steps)

class SOM_NeighborhoodFunctions:

    def gaussian_nbh(step,
                     mean_coordinates,
                     mapsize,
                     kernel_spread):
        
        euclidean_distance_map = np.full(mapsize, np.nan)
        iteration_map = np.nditer(euclidean_distance_map, flags = ['multi_index'])

        for _ in iteration_map:
            euclidean_distance = np.linalg.norm(np.array(mean_coordinates) -\
                                                np.array(iteration_map.multi_index))
            euclidean_distance_map[*iteration_map.multi_index] = euclidean_distance

        return np.exp(-(euclidean_distance_map**2)/(kernel_spread**2))