import numpy as np
from numpy import linalg as linalg

from mpdg_som_utils import chi_sq_dist
from mpdg_som_utils import training_step

from mpdg_som_utils import SOM_LearningRateFunctions as lrf
from mpdg_som_utils import SOM_NeighborhoodFunctions as nhb
from mpdg_som_utils import SOM_ErrorEstimators as e_est

#assumptions: data is not covariant

class SelfOrganizingMap:

    def __init__(self,
                 mapsize,
                 dimension: int = None,
                 name: str = 'som',
                 initialization: str = 'random',
                 learning_rate: float = 0.5,
                 maximum_steps: int = 1000,
                 ):
        
        self.name = name
        self.map_dimensionality = len(mapsize)
        self.step = 0

        if isinstance(maximum_steps, int) & (0 < maximum_steps):
            self.maximum_steps = maximum_steps

        else: raise(ValueError('The number of maximum steps must be a positive (non-zero) integer.'))

        self.use_covariance = False

        if isinstance(learning_rate, float) & (0 < learning_rate < 1):
            self.learning_rate = learning_rate
        
        else: raise(ValueError('The learning rate must be a float in the range (0, 1)'))

        if isinstance(mapsize, int) & (dimension != None):
            self.mapsize = [mapsize] * dimension

        elif isinstance(mapsize, int) & (dimension == None):
            raise(ValueError('Must also pass the number of dimensions of the map as an integer.'))

        elif isinstance(mapsize, list):
            self.mapsize = mapsize
        
        elif isinstance(mapsize, tuple):
            self.mapsize = [mapsize[i] for i in range(len(mapsize))]

        else: raise(ValueError('Mapsize must be an integer or a pair of numbers in list or tuple.'))

        if initialization in ['random', 'pca']:
            self.initialization = initialization
        
        else: raise(ValueError("Initialization type must be 'random' or 'pca'."))

    def load_data(self,
                  data,
                  variable_names: list = None):

        if len(np.shape(data)) == 2:
            self.data = np.array(data)

        elif (len(np.shape(data)) == 1) & (len(data) > 1):

            tuple_data = data.as_array()
            list_data = [list(values) for values in tuple_data]
            self.data = np.array(list_data)
        
        else: raise(TypeError('Please pass the data as a 2-d array. Each object should be an n-dimensional vector. All objects should have the same dimension.'))

        self.data_len = np.shape(self.data)[0]
        self.data_dim  = np.shape(self.data)[1]

        if variable_names != None:
            self.variable_names = variable_names
        
        elif variable_names == None:
            self.variable_names = [f'var{i}' for i in range(self.data_dim)]

        self.bmu_indices = np.full([self.data_len, self.map_dimensionality], 0, dtype = int)

    # def generate_normalization_matrix(self):

    #     normalization_matrix = {}

    #     for variable in self.variable_names:
    #         normalization_matrix[variable] = {}



    #     self.normalization_matrix = normalization_matrix

    # def normalize_data(self):

    #     self.normalization      

    def load_standard_deviations(self,
                                 stds):

        if len(np.shape(stds)) == 2:
            array_stds = np.array(stds)
            self.variances = array_stds**2

        elif (len(np.shape(stds)) == 1) & (len(stds) > 1):

            tuple_stds = stds.as_array()
            list_stds = [list(values) for values in tuple_stds]
            array_stds = np.array(list_stds)
            self.variances = array_stds**2
        
        else: raise(TypeError('Please pass the data as a 2-d array. Each object should be an n-dimensional vector. All objects should have the same dimension.'))

        self.use_covariance = True

    def data_statistics(self):

        print('| Data statistics ')
        print('stat\t', end = '')
        for i in range(self.data_dim): print(self.variable_names[i], end = '\t')
        print('\nmin\t', end = '')
        for i in range(self.data_dim): print(f'{np.min(self.data[:, i]):.3f}', end = '\t')
        print('\nmax\t', end = '')
        for i in range(self.data_dim): print(f'{np.max(self.data[:, i]):.3f}', end = '\t')
        print('\nmean\t', end = '')
        for i in range(self.data_dim): print(f'{np.mean(self.data[:, i]):.3f}', end = '\t')
        print('\nmedian\t', end = '')
        for i in range(self.data_dim): print(f'{np.median(self.data[:, i]):.3f}', end = '\t')
        print('\nstd\t', end = '')
        for i in range(self.data_dim): print(f'{np.std(self.data[:, i]):.3f}', end = '\t')

    def build_SOM(self):

        SOM_space = self.mapsize.copy()
        SOM_space.append(self.data_dim)

        if self.initialization == 'random':
            random_idx = np.random.rand(*self.mapsize) * self.data_len
            random_idx = np.array(random_idx, dtype = int)
            weights_map = self.data[random_idx]

        if self.initialization == 'pca':
            pca_covar_matrix = np.cov(self.data, rowvar = False)
            pca_eigvals, pca_eigvecs = np.linalg.eig(pca_covar_matrix)
            
            pca_indices = pca_eigvals.argsort()[-self.map_dimensionality:]
            pca_vectors = pca_eigvecs[pca_indices]

            principal_components = np.array([np.dot(self.data, pca_vector)\
                                             for pca_vector in pca_vectors])
            
            weights_map = np.full(SOM_space, np.nan)
            
            pca_map = np.meshgrid(*[np.linspace(np.min(principal_components[ii]),
                                                np.max(principal_components[ii]),
                                                self.mapsize[ii]) for ii in range(self.map_dimensionality)])

            for ii in range(self.data_dim):
                weights_map[..., ii] = np.sum([pca_map[jj] * pca_vectors[:, ii][jj]\
                                               for jj in range(self.map_dimensionality)], axis = 0)

        self.weights_map = weights_map

    def train(self,
              nans_in_empty_cells = False,
              error_func = None,
              error_thresh = None,
              debug_max_steps = 1):
        
        weights = self.weights_map.copy()
        complete_data = self.data.copy()
        complete_variance = self.variances.copy()

        error = 99
        while error > error_thresh:
            for index in range(self.data_len):
                weights, bmu_coords = training_step(weights,
                                                    complete_data[index],
                                                    complete_variance[index],
                                                    self.step,
                                                    lrf.power_law_lrf,
                                                    (1000, 0.5),
                                                    nhb.gaussian_nbh,
                                                    (self.mapsize, 2, 1000)
                                                    )
                self.bmu_indices[index] = (bmu_coords)
            
            error = e_est.max_misalignment(weights,
                                           complete_data,
                                           self.bmu_indices)
            print(f'Step {self.step} complete. Error: {error:.3f}')
            self.weights_map = weights
            self.step += 1
        
        print(f'SOM converged at step {self.step} to error {error}')

        # for step_count in range(debug_max_steps):
        #     for index in range(self.data_len):
        #         weights, bmu_coords = training_step(weights,
        #                                             complete_data[index],
        #                                             complete_variance[index],
        #                                             step_count,
        #                                             lrf.power_law_lrf,
        #                                             (1000, 0.5),
        #                                             nhb.gaussian_nbh,
        #                                             (self.mapsize, 2, 1000)
        #                                             )
        #         self.bmu_indices[index] = (bmu_coords)
        #     self.step += 1


    #next: build error estimator
    #after: build method for labelling pixels