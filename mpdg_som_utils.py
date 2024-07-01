import numpy as np
from numpy import linalg as linalg


def chi_sq_dist(weight_vector,
                data_vector,
                covar_vector = None,
                use_covariance = False,
                data_dim = None):

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

def find_bmu_coords(weight_vectors,
                    data_vector,
                    covar_vector):
    
    distance_map = np.full(np.shape(weight_vectors)[:-1], np.nan)
    iteration_map = np.nditer(distance_map, flags = ['multi_index'])

    for _ in iteration_map:
        distance = chi_sq_dist(weight_vectors[*iteration_map.multi_index],
                               data_vector,
                               covar_vector,
                               use_covariance = True)
        distance_map[*iteration_map.multi_index] = distance

    argmin_idx = np.argmin(distance_map)
    return np.unravel_index(argmin_idx, np.shape(weight_vectors)[:-1])


def training_step(weight_vectors,
                  data_vector,
                  covar_vector,
                  step,
                  learning_rate_function,
                  lrf_args,
                  neighborhood_function,
                  nbh_args
                  ):
    
    current_weight_vectors = weight_vectors.copy()

    bmu_coords = find_bmu_coords(current_weight_vectors,
                                 data_vector,
                                 covar_vector)

    learning_rate_mult = learning_rate_function(step,
                                                *lrf_args)

    neighborhood_mult = neighborhood_function(step, bmu_coords,
                                              *nbh_args)
    neighborhood_mult = np.stack([neighborhood_mult] * len(data_vector), axis = -1)

    data_vector_map = np.full(np.shape(current_weight_vectors),
                              data_vector)

    updated_weight_vectors = current_weight_vectors +\
                             learning_rate_mult * neighborhood_mult *\
                            (data_vector_map - current_weight_vectors)
    
    return updated_weight_vectors, bmu_coords

class SOM_LearningRateFunctions:

    def power_law_lrf(step,
                  maximum_steps, learning_rate):
        
        return learning_rate ** (step/maximum_steps)

class SOM_NeighborhoodFunctions:

    def gaussian_nbh(step,
                     mean_coordinates,
                     mapsize,
                     init_kernel_spread,
                     maximum_steps):
        
        euclidean_distance_map = np.full(mapsize, np.nan)
        iteration_map = np.nditer(euclidean_distance_map, flags = ['multi_index'])

        for _ in iteration_map:
            euclidean_distance = np.linalg.norm(np.array(mean_coordinates) -\
                                                np.array(iteration_map.multi_index))
            euclidean_distance_map[*iteration_map.multi_index] = euclidean_distance

        kernel_spread = init_kernel_spread ** (1 - (step/maximum_steps))

        return np.exp(-(euclidean_distance_map**2)/(kernel_spread**2))
    
class SOM_ErrorEstimators:

    def max_misalignment(weight_vectors,
                         data,
                         bmu_indices):

        max_misalign = 0
        for index in range(len(data)):
            bmu_index = bmu_indices[index]
            misalign = np.dot(data[index], weight_vectors[*bmu_index, :])
            if misalign > max_misalign: max_misalign = misalign
        
        return max_misalign

    def quantization_error(step):
        raise(NotImplementedError())
