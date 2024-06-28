import numpy as np
from numpy import linalg as linalg

from mpdg_som_utils import chi_sq_dist
from mpdg_som_utils import training_step

import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 24,
    'axes.labelsize': 'large',
    'mathtext.fontset': 'stix'
})

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

        if self.map_dimensionality > self.data_dim:
            raise(AssertionError('The map cannot have more dimensions than the data.'))

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
        while self.step < self.maximum_steps:
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

    def label_map(self,
                  parameters,
                  parameter_names = None):
        
        if len(np.shape(parameters)) == 2:
            self.parameters = np.array(parameters)

        elif (len(np.shape(parameters)) == 1) & (len(parameters)> 1):

            tuple_params = parameters.as_array()
            list_params = [list(values) for values in tuple_params]
            self.parameters = np.array(list_params)

        if np.shape(self.parameters)[0] != self.data_len:
            raise(AssertionError('The number of parameter data points does not match the number of data points used to build the map!'))

        self.params_dim = np.shape(self.parameters)[1]

        if parameter_names != None:
            self.parameter_names = parameter_names
        
        elif parameter_names == None:
            self.parameter_names = [f'param{i}' for i in range(self.params_dim)]

        populated_cells = np.unique(self.bmu_indices, axis = 0)

        if len(populated_cells) > np.prod(self.mapsize):
            raise(AssertionError('There are more populated cells than there are cells in the entire map. Check logic.'))

        self.map_labels = np.full([*self.mapsize, self.params_dim], np.nan)

        for cell in populated_cells:

            matching_idx = np.all(self.bmu_indices == cell, axis = -1)
            self.map_labels[*cell] = np.median(self.parameters[matching_idx], axis = 0)

    def show_map(self, show_labeled = False,
                 cmap = 'jet_r'):

        if self.map_dimensionality != 2:
            raise(NotImplementedError('The module can only display 2-d maps for now.'))

        print(f'\n| SOM. Step {self.step}. Initialization: {self.initialization}')


        if not show_labeled:

            fig = plt.figure(figsize = (self.data_dim * 5, 10), constrained_layout = True)
            
            for i, name in enumerate(self.variable_names):
                ax = fig.add_subplot(1, self.data_dim, i + 1)
                imsh = ax.imshow(self.weights_map[..., i], origin = 'lower', cmap = cmap)
                ax.axis('off')
                fig.colorbar(mappable = imsh, ax = ax,
                            label = name, location = 'bottom', pad = 0.01)

        if show_labeled:

            variables_dim = np.shape(self.variable_names)[1]
            fig = plt.figure(figsize = (self.variables_dim * 5, 10), constrained_layout = True)
            
            for i, name in enumerate(self.parameter_names):
                ax = fig.add_subplot(1, variables_dim, i + 1)
                imsh = ax.imshow(self.map_labels[..., i], origin = 'lower', cmap = cmap)
                ax.axis('off')
                fig.colorbar(mappable = imsh, ax = ax,
                             label = name, location = 'bottom', pad = 0.01)
                
    def predict(self,
                prediction_input):
        
        raise(NotImplementedError)
                
    def prediction_statistics(self,
                              return_values = False):
        
                        
        raise(NotImplementedError)