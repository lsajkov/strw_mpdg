import numpy as np
from numpy import linalg as linalg

from mpdg_som_utils import find_bmu_coords
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
from mpdg_som_utils import SOM_Normalizers as som_norm

#assumptions: data is not covariant

class SelfOrganizingMap:

    def __init__(self,
                 name: str = '',
                 mapsize = None,
                 dimension: int = None,
                 initialization: str = 'pca',
                 termination: str = 'maximum_steps',
                 learning_rate_function: str = 'power_law',
                 neighborhood_function: str = 'gaussian',
                 error_estimator: str = 'maximum_misalignment',
                 normalization: str = 'unit_range',
                 learning_rate: float = 0.5,
                 kernel_spread: float = 1.,
                 maximum_steps: int = 100,
                 error_thresh: float = None
                 ):
        
        self.name = name

        if isinstance(mapsize, int) & (dimension != None):
            self.mapsize = [mapsize] * dimension
        elif isinstance(mapsize, int) & (dimension == None):
            raise(ValueError('Must also pass the number of dimensions of the map as an integer.'))
        elif isinstance(mapsize, list):
            self.mapsize = mapsize
        elif isinstance(mapsize, tuple):
            self.mapsize = [mapsize[i] for i in range(len(mapsize))]
        else: raise(ValueError('Mapsize must be an integer or a pair of numbers in list or tuple.'))

        self.map_dimensionality = len(self.mapsize)
                    
        if initialization in ['random', 'pca']:
            self.initialization = initialization
        else: raise(TypeError('Please pass an initalization type with initializaton = "type". Types can be: "random", "pca".'))

        if termination in ['error_thresh', 'maximum_steps', 'either']:
            self.termination = termination
        else: raise(TypeError('Please pass a termination type with termination = "type". Types can be: "error_thresh", "maximum_steps", "either".'))

        if learning_rate_function in ['power_law']:
            self.learning_rate_function = learning_rate_function
        else: raise(TypeError('PSOM.mapsizeease pass a learning rate function with learning_rate_function = "type". Types can be: "power_law".'))

        if neighborhood_function in ['gaussian']:
            self.neighborhood_function = neighborhood_function
        else: raise(TypeError('Please pass a neighborhood function with neighborhood_function = "type". Types can be: "gaussian".'))

        if ((termination == 'maximum_steps') | (termination == 'either')) & (error_estimator in ['quantization_error', 'maximum_misalignment']):
            self.error_estimator = error_estimator
        elif termination == 'error_thresh':
            raise(TypeError('Please pass an error estimator with error_estimator = "type". Types can be: "mean_misalignment", "maximum_misalignment".'))

        if isinstance(learning_rate, float) & (0 < learning_rate < 1):
            self.learning_rate = learning_rate
        else: raise(ValueError('The learning rate must be a float in the range (0, 1)'))

        if termination != 'maximum_steps': self.maximum_steps = maximum_steps
        elif ((termination == 'maximum_steps') | (termination == 'either')) & (maximum_steps > 0) & isinstance(maximum_steps, int):
            self.maximum_steps = maximum_steps
        else: raise(ValueError('The number of maximum steps must be a positive (non-zero) integer.'))

        if (termination != 'error_thresh') & (termination != 'either'): self.error_thresh = 0
        elif ((termination == 'error_thresh') | (termination == 'either')) & (error_thresh != None):
            self.error_thresh = error_thresh
        elif ((termination == 'error_thresh') | (termination == 'either')) & (error_thresh == None):
            raise(ValueError('Please pass an error threshold to use error_thresh or either as a terminator.'))
        
        self.kernel_spread = kernel_spread
        self.normalization = normalization
        self.use_covariance = False


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

        # self.randomized_data_indices = np.arange(0, self.data_len, 1)
        self.bmu_indices = np.full([self.data_len, self.map_dimensionality], 0, dtype = int)


    def normalize_data(self):

        if self.normalization in ['zero_mean_unit_variance', 'zmuv']:
            normalizer = som_norm.ZeroMean_UnitVariance(self.variable_names)
        
        if self.normalization in ['unit_range', 'ur']:
            normalizer = som_norm.UnitRange(self.variable_names)
            
        original_data = self.data.copy()
        normalized_data = normalizer.normalize(original_data)
        self.data = normalized_data
        self.normalization_params = normalizer.normalization_params
            

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


    def normalize_standard_deviations(self):

        original_stds = np.sqrt(self.variances.copy())
        normalized_stds = original_stds.copy()
        if self.normalization in ['zero_mean_unit_variance', 'zmuv']:
            for i, variable in enumerate(self.variable_names):
                normalized_stds[:, i] -= self.normalization_params[variable]['mean']
                normalized_stds[:, i] /= self.normalization_params[variable]['std']

        if self.normalization in ['unit_range', 'ur']:
            for i, variable in enumerate(self.variable_names):
                # normalized_stds[:, i] -= self.normalization_params[variable]['min']
                normalized_stds[:, i] /= self.normalization_params[variable]['max'] - self.normalization_params[variable]['min']

        self.variances = normalized_stds**2

    def load_labeling_data(self,
                           labeling_data,
                           parameter_names: list = None):

        if len(np.shape(labeling_data)) == 2:
            self.labeling_data = np.array(labeling_data)

        elif (len(np.shape(labeling_data)) == 1) & (len(labeling_data) > 1):

            tuple_labeling_data = labeling_data.as_array()
            list_labeling_data = [list(values) for values in tuple_labeling_data]
            self.labeling_data = np.array(list_labeling_data)
        
        else: raise(TypeError('Please pass the data as a 2-d array. Each object should be an n-dimensional vector. All objects should have the same dimension.'))

        self.labeling_data_len = np.shape(self.labeling_data)[0]
        self.labeling_data_dim  = np.shape(self.labeling_data)[1]
        self.labels_dim = self.labeling_data_dim - self.data_dim 

        if parameter_names != None:
            self.parameter_names = parameter_names
        
        elif parameter_names == None:
            self.parameter_names = [f'var{i}' for i in range(self.labeling_data_dim - self.data_dim)]

    def normalize_labeling_data(self):

        if self.normalization in ['zero_mean_unit_variance', 'zmuv']:
            normalizer = som_norm.ZeroMean_UnitVariance(self.variable_names)
        
        if self.normalization in ['unit_range', 'ur']:
            for i, var in enumerate(self.normalization_params):
                self.labeling_data[:, i] -= self.normalization_params[var]['min']
                self.labeling_data[:, i] /= self.normalization_params[var]['max'] - self.normalization_params[var]['min']

    def load_labeling_standard_deviations(self,
                                          label_stds):

        if len(np.shape(label_stds)) == 2:
            array_label_stds = np.array(label_stds)
            self.label_variances = array_label_stds**2

        elif (len(np.shape(label_stds)) == 1) & (len(label_stds) > 1):

            tuple_label_stds = label_stds.as_array()
            list_label_stds = [list(values) for values in tuple_label_stds]
            array_label_stds = np.array(list_label_stds)
            self.label_variances = array_label_stds**2
        
        else: raise(TypeError('Please pass the data as a 2-d array. Each object should be an n-dimensional vector. All objects should have the same dimension.'))

    def normalize_labeling_standard_deviations(self):

        original_label_stds = np.sqrt(self.label_variances.copy())
        normalized_label_stds = original_label_stds.copy()
        if self.normalization in ['zero_mean_unit_variance', 'zmuv']:
            for i, variable in enumerate(self.variable_names):
                normalized_label_stds[:, i] -= self.normalization_params[variable]['mean']
                normalized_label_stds[:, i] /= self.normalization_params[variable]['std']

        if self.normalization in ['unit_range', 'ur']:
            for i, variable in enumerate(self.variable_names):
                normalized_label_stds[:, i] /= self.normalization_params[variable]['max'] - self.normalization_params[variable]['min']

        self.label_variances = normalized_label_stds**2

    def data_statistics(self):

        print('| Data statistics ')
        print(f'total entries: {self.data_len}\n')
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
            pca_data = self.data.copy()
            randomized_idx = np.arange(0, self.data_len, 1)
            np.random.shuffle(randomized_idx)
            pca_data = pca_data[randomized_idx]

            pca_covar_matrix = np.cov(pca_data, rowvar = False)
            pca_eigvals, pca_eigvecs = np.linalg.eig(pca_covar_matrix)
            
            pca_indices = pca_eigvals.argsort()[-self.map_dimensionality:]
            pca_vectors = pca_eigvecs[pca_indices]

            principal_components = np.array([np.dot(pca_data, pca_vector)\
                                             for pca_vector in pca_vectors])
            
            weights_map = np.full(SOM_space, np.nan)
            
            pca_map = np.meshgrid(*[np.linspace(np.min(principal_components[ii]),
                                                np.max(principal_components[ii]),
                                                self.mapsize[ii]) for ii in range(self.map_dimensionality)])

            for ii in range(self.data_dim):
                weights_map[..., ii] = np.sum([pca_map[jj] * pca_vectors[:, ii][jj]\
                                               for jj in range(self.map_dimensionality)], axis = 0).transpose()

        self.weights_map = weights_map
        self.step = 0


    def train(self):
        
        self.errors = []
        
        weights = self.weights_map.copy()
        complete_data = self.data.copy()
        complete_variance = self.variances.copy() if self.use_covariance else np.ones_like(self.data)

        indexed_map = np.empty(self.mapsize,
                               dtype = object)

        indexed_map = np.empty(self.mapsize,
                               dtype = object)

        self.randomized_data_indices = np.arange(0, self.data_len)
        
        continue_training = True
        while continue_training: 

            for index, _ in np.ndenumerate(indexed_map):
                indexed_map[index] = []
            self.indexed_map = indexed_map

            for index, _ in np.ndenumerate(indexed_map):
                indexed_map[index] = []
            self.indexed_map = indexed_map

            np.random.shuffle(self.randomized_data_indices)

            print(f'Step {self.step + 1} [{30 * " "}] 0/{self.data_len}', end = '\r')
            for i, index in enumerate(self.randomized_data_indices):
                weights, bmu_coords = training_step(weights,
                                                    complete_data[index],
                                                    complete_variance[index],
                                                    self.step,
                                                    lrf.power_law_lrf,
                                                    (self.learning_rate, self.maximum_steps),
                                                    nhb.gaussian_nbh,
                                                    (self.mapsize, self.kernel_spread, self.maximum_steps)
                                                    )
                self.bmu_indices[index] = (bmu_coords)
                self.indexed_map[bmu_coords].append(index)
                print(f'Step {self.step + 1} [{int(30 * i / self.data_len) * "*"}{(30 - int(30 * i / self.data_len)) * " "}] {i}/{self.data_len}',
                    end = '\r')
            
            self.weights_map = weights
            self.step += 1

            error = e_est.quantization_error(weights,
                                             complete_data,
                                             self.bmu_indices)
            self.errors.append(error)
            self.error = error

            if self.step >= self.maximum_steps or self.error <= self.error_thresh:
                if self.termination == 'either' or\
                  (self.termination == 'maxium_steps' and self.step >= self.maximum_steps) or\
                  (self.termination == 'error_thresh' and self.error <= self.error_thresh):
                    continue_training = False

            # if self.step >= self.maximum_steps:
            #     if self.termination == 'either': continue_training = False
            #     elif self.termination == 'maximum_steps': continue_training = False
            
            # if self.error <= self.error_thresh:
            #     if self.termination == 'either': continue_training = False
            #     elif self.termination == 'error_thresh': continue_training = False
            
            print(f'Step {self.step} complete. Error: {self.error:.3f}                                   ')

        print(f'SOM converged at step {self.step} to error {self.error}')
        return error

    # def label_map(self):
        
    #     labeled_map_values = np.empty([*self.mapsize, self.labels_dim, 2],
    #                                 dtype = object)
    #     for index, _ in np.ndenumerate(labeled_map_values):
    #         labeled_map_values[index] = []

    def label_map(self):
        
        labeled_map_values = np.empty([*self.mapsize, self.labels_dim, 2],
                                    dtype = object)
        for index, _ in np.ndenumerate(labeled_map_values):
            labeled_map_values[index] = []

        for index in range(self.labeling_data_len):

            bmu_coords = find_bmu_coords(self.weights_map,
                                        self.labeling_data[index, :self.data_dim],
                                        self.label_variances[index, :self.data_dim])

            for i in range(self.labels_dim):
                labeled_map_values[*bmu_coords, i, 0].append(self.labeling_data[index, self.data_dim + i])
                labeled_map_values[*bmu_coords, i, 1].append(self.label_variances[index,  self.data_dim + i]) 

        labeled_map = np.empty([*self.mapsize, self.labels_dim, 2])

        for i in range(self.labels_dim):
            for cell, _ in np.ndenumerate(labeled_map[..., 0, 0]):
                if len(labeled_map_values[*cell, i, 0]) > 0:
                    mean, insqvar = np.average(labeled_map_values[*cell, i, 0],
                                                weights = 1/np.array(labeled_map_values[*cell, i, 1])**2,
                                                returned = True)
                    labeled_map[*cell, i, 0] = mean
                    labeled_map[*cell, i, 1] = np.sqrt(1/insqvar)
                else: labeled_map[*cell, i] = [np.nan, np.nan]

        self.labeled_map_values = labeled_map_values
        self.labeled_map = labeled_map

    def produce_occupancy_map(self,
                              show_map = False, cmap = 'jet', vmin = 0, vmax = 100,
                              save_fig = False, save_path = '',
                              save_map = False):

        occupancy_map = np.full(self.mapsize, np.nan)

        for cell, values in np.ndenumerate(self.labeled_map_values[..., 0, 0]):
            if len(values) > 0:
                occupancy_map[cell] = len(values)

        if show_map:

            fig = plt.figure(figsize = (9.6, 8), constrained_layout = True)
            ax = fig.add_subplot()
            occ_mp = ax.imshow(occupancy_map,
                origin = 'lower', cmap = cmap,
                vmin = vmin, vmax = vmax)
            ax.axis('off')

            fig.colorbar(mappable = occ_mp, location = 'bottom', fraction = 1, shrink = 0.66, pad = 0.005,
                         label = f'Occupancy (galaxy/pixel)\nMean occupancy: {np.nanmean(occupancy_map):.1f}')
            
            if save_fig:
                fig.savefig(f'{save_path}/SOM_occupancy_map')

        if save_map:
            np.save(f'{save_path}/SOM_occupancy_map',
                    occupancy_map, allow_pickle = True)

    def produce_quality_map(self,
                            show_map = False, cmap = 'jet',
                            save_fig = False, save_path = '',
                            save_map = False):

        quality_map = np.full([*self.mapsize, self.labeling_data_dim - self.data_dim], np.nan)

        for cell, _ in np.ndenumerate(quality_map[..., 0]):
            for i in range(self.labeling_data_dim - self.data_dim):
                try:
                    param_phot = np.mean(self.data[self.indexed_map[*cell], self.data_dim - 1 + i])
                    param_spec = np.mean(self.labeled_map_values[*cell, i, 0])

                    quality_map[*cell, i] = np.abs(param_spec - param_phot)
                except:
                    continue

        if show_map:

            fig = plt.figure(figsize = (10, 5 * (self.labeling_data_dim - self.data_dim)),
                             constrained_layout = True)
            
            for i in range(self.labeling_data_dim - self.data_dim):
                ax = fig.add_subplot(1, self.labeling_data_dim - self.data_dim, i + 1)

                qual_mp = ax.imshow(quality_map[..., i],
                    origin = 'lower', cmap = cmap)
                ax.axis('off')

                fig.colorbar(mappable = qual_mp, location = 'bottom', fraction = 1, shrink = 0.66, pad = 0.005,
                            label = self.parameter_names[i])
            
        #     if save_fig:
        #         fig.savefig(f'{save_path}/SOM_quality_map')

        if save_map:
            np.save(f'{save_path}/SOM_quality_map',
                    quality_map, allow_pickle = True)

        self.quality_map = quality_map 

    def show_map(self, show_labeled = False,
                 cmap = 'jet_r', log_norm = [],
                 save = False, save_path = None,
                 vmin = 0, vmax = 250):

        if self.map_dimensionality != 2:
            raise(NotImplementedError('The module can only display 2-d maps for now.'))

        print(f'\n| SOM. Step {self.step}. Initialization: {self.initialization}')

        if save_path == None:
            save_path = './SOM_map'

        if not show_labeled:

            fig = plt.figure(figsize = (self.data_dim * 5, 10), constrained_layout = True)
            
            for i, name in enumerate(self.variable_names):
                ax = fig.add_subplot(1, self.data_dim, i + 1)

                if name in log_norm:
                    imsh = ax.imshow(np.log10(self.weights_map[..., i]), origin = 'lower', cmap = cmap)
                else:
                    imsh = ax.imshow(self.weights_map[..., i], origin = 'lower', cmap = cmap)
                    
                ax.axis('off')
                fig.colorbar(mappable = imsh, ax = ax,
                            label = name, location = 'bottom', pad = 0.01)

        if show_labeled:

            fig = plt.figure(figsize = (self.labels_dim * 5, 10), constrained_layout = True)
            
            for i, name in enumerate(self.parameter_names):
                ax = fig.add_subplot(1, self.labels_dim, i + 1)

                if name in log_norm:
                    imsh = ax.imshow(np.log10(self.labeled_map[..., i, 0]), origin = 'lower', cmap = cmap)
                else:
                    imsh = ax.imshow(self.labeled_map[..., i, 0], origin = 'lower', cmap = cmap)
                # imsh = ax.imshow(self.labeled_map[..., i, 0],
                #                  origin = 'lower', cmap = cmap)
                # imsh = ax.imshow(np.sum(self.distribution_xs[i] * self.labeled_map[..., i, :],
                #                         axis = -1),
                #                  origin = 'lower', cmap = cmap)
                ax.axis('off')
                fig.colorbar(mappable = imsh, ax = ax,
                             label = name, location = 'bottom', pad = 0.01)

        if save:
            fig.savefig(save_path,
                        bbox_inches = 'tight')

    def predict(self,
                prediction_input,
                prediction_stds = None):
        
        self.prediction_stds = prediction_stds
        
        if len(np.shape(prediction_input)) == 2:
            self.prediction_input = np.array(prediction_input)

        elif (len(np.shape(prediction_input)) == 1) & (len(prediction_input) > 1):

            tuple_input = prediction_input.as_array()
            list_input = [list(values) for values in tuple_input]
            self.prediction_input = np.array(list_input)
            
        else: raise(TypeError('Please pass the data as a 2-d array. Each object should be an n-dimensional vector. All objects should have the same dimension.'))

        prediction_input_len, prediction_input_dim = np.shape(self.prediction_input)
        
        if prediction_input_dim != self.data_dim:
            raise(AssertionError('The dimension of the prediction data does not match the dimension of the data used to train the SOM.'))

        prediction_results = np.full([prediction_input_len, self.labeling_data_dim - self.data_dim], np.nan)
        prediction_sigmas  = np.full([prediction_input_len, self.labeling_data_dim - self.data_dim], np.nan)

        for index in range(prediction_input_len):

            bmu_coords = find_bmu_coords(self.weights_map,
                                         self.prediction_input[index],
                                         self.prediction_stds[index])
            prediction_results[index] = self.labeled_map[*bmu_coords, :, 0]
            prediction_sigmas[index]  = self.labeled_map[*bmu_coords, :, 1]

            prediction_results[index] = self.labeled_map[*bmu_coords, :, 0]
            prediction_sigmas[index]  = self.labeled_map[*bmu_coords, :, 1]

        self.prediction_results = prediction_results
        self.prediction_sigmas  = prediction_sigmas

        self.prediction_sigmas  = prediction_sigmas

    


    def save_outputs(self,
                     directory_path,
                     save_weights = True,
                     save_predictions = False,
                     save_parameters = False):
        
        if save_weights:
            np.save(f'{directory_path}/SOM_weights', self.weights_map)

        if save_parameters:
            map_parameters = {}
            map_parameters['name']                   = self.name                  
            map_parameters['mapsize']                = self.mapsize            
            map_parameters['initialization']         = self.initialization        
            map_parameters['termination']            = self.termination           
            map_parameters['learning_rate_function'] = self.learning_rate_function
            map_parameters['neighborhood_function']  = self.neighborhood_function 
            map_parameters['error_estimator']        = self.error_estimator       
            map_parameters['learning_rate']          = self.learning_rate         
            map_parameters['kernel_spread']          = self.kernel_spread         
            map_parameters['maximum_steps']          = self.maximum_steps         
            map_parameters['error_thresh']           = self.error_thresh          
             
            np.save(f'{directory_path}/SOM_params', map_parameters)

    # def label_map(self,
    #               sigmas_data: list = None,
    #               sigmas_add: list = None,
    #               pdr: int = 1000,):
        
    #     self.pdr = pdr
    #     #pdr = probability density resolution

    #     #initialize the labeled map as a probability distribution for each variable in each cell
    #     #the complete map will have (number of variables * number of cells) probability distributions
    #     #the distributions are initialized with p(var|obs) = 0.0, where var is the given variable and obs is an observed value
    #     labeled_map = np.zeros([*self.mapsize, self.labels_dim, pdr])

    #     #create the viable distribution space. This is essentially the integration bounds where var can be non-zero
    #     distribution = np.array([np.linspace(np.min(self.labeling_data[:, i]), np.max(self.labeling_data[:, i]), pdr)\
    #                             for i in range(self.data_dim, self.labeling_data_dim)])

    #     #in each probability distribution, add a delta function centered at the
    #     #observed value of the variable
    #     for index in range(self.labeling_data_len):

    #         bmu_coords = find_bmu_coords(self.weights_map,
    #                                      self.labeling_data[index, :self.data_dim],
    #                                      self.label_variances[index, :self.data_dim])

    #         for i in range(self.labels_dim):
    #             delta_index = np.abs(distribution[i] - self.labeling_data[index, self.data_dim + i]).argmin()
    #             labeled_map[*bmu_coords, i, delta_index] += 1

    #     def gaussian(dist,
    #                  sigma_data,
    #                  sigma_add, N_cell):
            
    #         sigma = np.sqrt((sigma_data ** 2) * (1 - 1/A_c) + (sigma_add ** 2)/N_cell)

    #         return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-((dist - np.mean(dist))**2)/ (2 * sigma ** 2))

    #     sigmas_data = sigmas_data or [0.1, 0.05]
    #     sigmas_add = sigmas_add or [0, 0]
    #     # if sigmas_data == None: sigmas_data = [0.2, 0.05]
    #     # if sigmas_add == None: sigmas_add = [0.9, 0.05]

    #     iteration_map = np.nditer(np.full(self.mapsize, 0), flags = ['multi_index'])
    #     for _ in iteration_map:
    #         for i in range(self.labels_dim):
                
    #             A_c = np.sum(labeled_map[iteration_map.multi_index][i])
    #             labeled_map[iteration_map.multi_index][i] /= A_c

    #             convolved_distribution = np.convolve(labeled_map[iteration_map.multi_index][i],
    #                                                     gaussian(distribution[i],
    #                                                             sigmas_data[i], sigmas_add[i],
    #                                                             A_c), mode = 'same')
    #             convolved_distribution /= np.sum(convolved_distribution)

    #             labeled_map[iteration_map.multi_index][i] = convolved_distribution
    #             # if np.sum(labeled_map[iteration_map.multi_index][i]) == 0.:
    #             #     continue

    #             # else: 

    #     # finally, set empty cells to all-nans
    #     # for i in range(self.labels_dim):
    #     #     empty_cells = 
    #     # iteration_map = np.nditer(np.full(self.mapsize, 0), flags = ['multi_index'])
    #     # for _ in iteration_map:
    #     #     for i in range(self.labels_dim):
    #     #         if np.sum(labeled_map[*iteration_map.multi_index][i]) == 0.:
    #     #             labeled_map[*iteration_map.multi_index][i] = np.full(pdr, np.nan)

    #     self.distribution_xs = distribution
    #     self.labeled_map = labeled_map

    # def show_map(self, show_labeled = False,
    #              cmap = 'jet_r',
    #              save = False, save_path = None):

    #     if self.map_dimensionality != 2:
    #         raise(NotImplementedError('The module can only display 2-d maps for now.'))

    #     print(f'\n| SOM. Step {self.step}. Initialization: {self.initialization}')

    #     if save_path == None:
    #         save_path = './SOM_map'

    #     if not show_labeled:

    #         fig = plt.figure(figsize = (self.data_dim * 5, 10), constrained_layout = True)
            
    #         for i, name in enumerate(self.variable_names):
    #             ax = fig.add_subplot(1, self.data_dim, i + 1)
    #             imsh = ax.imshow(self.weights_map[..., i], origin = 'lower', cmap = cmap)
    #             ax.axis('off')
    #             fig.colorbar(mappable = imsh, ax = ax,
    #                         label = name, location = 'bottom', pad = 0.01)

    #     if show_labeled:

    #         fig = plt.figure(figsize = (self.labels_dim * 5, 10), constrained_layout = True)
            
    #         for i, name in enumerate(self.parameter_names):
    #             ax = fig.add_subplot(1, self.labels_dim, i + 1)
    #             imsh = ax.imshow(np.sum(self.distribution_xs[i] * self.labeled_map[..., i, :],
    #                                     axis = -1),
    #                              origin = 'lower', cmap = cmap)
    #             ax.axis('off')
    #             fig.colorbar(mappable = imsh, ax = ax,
    #                          label = name, location = 'bottom', pad = 0.01)

    #     if save:
    #         fig.savefig(save_path,
    #                     bbox_inches = 'tight')

    # def label_map(self,
    #               sigmas_data: list = None,
    #               sigmas_add: list = None,
    #               pdr: int = 1000,):
        
    #     self.pdr = pdr
    #     #pdr = probability density resolution

    #     #initialize the labeled map as a probability distribution for each variable in each cell
    #     #the complete map will have (number of variables * number of cells) probability distributions
    #     #the distributions are initialized with p(var|obs) = 0.0, where var is the given variable and obs is an observed value
    #     labeled_map = np.zeros([*self.mapsize, self.labels_dim, pdr])

    #     #create the viable distribution space. This is essentially the integration bounds where var can be non-zero
    #     distribution = np.array([np.linspace(np.min(self.labeling_data[:, i]), np.max(self.labeling_data[:, i]), pdr)\
    #                             for i in range(self.data_dim, self.labeling_data_dim)])

    #     #in each probability distribution, add a delta function centered at the
    #     #observed value of the variable
    #     for index in range(self.labeling_data_len):

    #         bmu_coords = find_bmu_coords(self.weights_map,
    #                                      self.labeling_data[index, :self.data_dim],
    #                                      self.label_variances[index, :self.data_dim])

    #         for i in range(self.labels_dim):
    #             delta_index = np.abs(distribution[i] - self.labeling_data[index, self.data_dim + i]).argmin()
    #             labeled_map[*bmu_coords, i, delta_index] += 1

    #     def gaussian(dist,
    #                  sigma_data,
    #                  sigma_add, N_cell):
            
    #         sigma = np.sqrt((sigma_data ** 2) * (1 - 1/A_c) + (sigma_add ** 2)/N_cell)

    #         return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-((dist - np.mean(dist))**2)/ (2 * sigma ** 2))

    #     sigmas_data = sigmas_data or [0.1, 0.05]
    #     sigmas_add = sigmas_add or [0, 0]
    #     # if sigmas_data == None: sigmas_data = [0.2, 0.05]
    #     # if sigmas_add == None: sigmas_add = [0.9, 0.05]

    #     iteration_map = np.nditer(np.full(self.mapsize, 0), flags = ['multi_index'])
    #     for _ in iteration_map:
    #         for i in range(self.labels_dim):
                
    #             A_c = np.sum(labeled_map[iteration_map.multi_index][i])
    #             labeled_map[iteration_map.multi_index][i] /= A_c

    #             convolved_distribution = np.convolve(labeled_map[iteration_map.multi_index][i],
    #                                                     gaussian(distribution[i],
    #                                                             sigmas_data[i], sigmas_add[i],
    #                                                             A_c), mode = 'same')
    #             convolved_distribution /= np.sum(convolved_distribution)

    #             labeled_map[iteration_map.multi_index][i] = convolved_distribution
    #             # if np.sum(labeled_map[iteration_map.multi_index][i]) == 0.:
    #             #     continue

    #             # else: 

    #     # finally, set empty cells to all-nans
    #     # for i in range(self.labels_dim):
    #     #     empty_cells = 
    #     # iteration_map = np.nditer(np.full(self.mapsize, 0), flags = ['multi_index'])
    #     # for _ in iteration_map:
    #     #     for i in range(self.labels_dim):
    #     #         if np.sum(labeled_map[*iteration_map.multi_index][i]) == 0.:
    #     #             labeled_map[*iteration_map.multi_index][i] = np.full(pdr, np.nan)

    #     self.distribution_xs = distribution
    #     self.labeled_map = labeled_map

    # def show_map(self, show_labeled = False,
    #              cmap = 'jet_r',
    #              save = False, save_path = None):

    #     if self.map_dimensionality != 2:
    #         raise(NotImplementedError('The module can only display 2-d maps for now.'))

    #     print(f'\n| SOM. Step {self.step}. Initialization: {self.initialization}')

    #     if save_path == None:
    #         save_path = './SOM_map'

    #     if not show_labeled:

    #         fig = plt.figure(figsize = (self.data_dim * 5, 10), constrained_layout = True)
            
    #         for i, name in enumerate(self.variable_names):
    #             ax = fig.add_subplot(1, self.data_dim, i + 1)
    #             imsh = ax.imshow(self.weights_map[..., i], origin = 'lower', cmap = cmap)
    #             ax.axis('off')
    #             fig.colorbar(mappable = imsh, ax = ax,
    #                         label = name, location = 'bottom', pad = 0.01)

    #     if show_labeled:

    #         fig = plt.figure(figsize = (self.labels_dim * 5, 10), constrained_layout = True)
            
    #         for i, name in enumerate(self.parameter_names):
    #             ax = fig.add_subplot(1, self.labels_dim, i + 1)
    #             imsh = ax.imshow(np.sum(self.distribution_xs[i] * self.labeled_map[..., i, :],
    #                                     axis = -1),
    #                              origin = 'lower', cmap = cmap)
    #             ax.axis('off')
    #             fig.colorbar(mappable = imsh, ax = ax,
    #                          label = name, location = 'bottom', pad = 0.01)

    #     if save:
    #         fig.savefig(save_path,
    #                     bbox_inches = 'tight')

    # def label_map(self,
    #               parameters,
    #               parameter_names = None):
        
    #     if len(np.shape(parameters)) == 2:
    #         self.parameters = np.array(parameters)

    #     elif (len(np.shape(parameters)) == 1) & (len(parameters)> 1):

    #         tuple_params = parameters.as_array()
    #         list_params = [list(values) for values in tuple_params]
    #         self.parameters = np.array(list_params)

    #     if np.shape(self.parameters)[0] != self.data_len:
    #         raise(AssertionError('The number of parameter data points does not match the number of data points used to build the map!'))

    #     self.params_dim = np.shape(self.parameters)[1]

    #     if parameter_names != None:
    #         self.parameter_names = parameter_names
        
    #     elif parameter_names == None:
    #         self.parameter_names = [f'param{i}' for i in range(self.params_dim)]

    #     populated_cells = np.unique(self.bmu_indices, axis = 0)

    #     if len(populated_cells) > np.prod(self.mapsize):
    #         raise(AssertionError('There are more populated cells than there are cells in the entire map. Check logic.'))

    #     self.map_labels = np.full([*self.mapsize, self.params_dim], np.nan)

    #     for cell in populated_cells:

    #         matching_idx = np.all(self.bmu_indices == cell, axis = -1)
    #         self.map_labels[*cell] = np.median(self.parameters[matching_idx], axis = 0)

    # def predict(self,
    #             prediction_input,
    #             prediction_stds = None):
        
    #     self.prediction_stds = prediction_stds
        
    #     if len(np.shape(prediction_input)) == 2:
    #         self.prediction_input = np.array(prediction_input)

    #     elif (len(np.shape(prediction_input)) == 1) & (len(prediction_input) > 1):

    #         tuple_input = prediction_input.as_array()
    #         list_input = [list(values) for values in tuple_input]
    #         self.prediction_input = np.array(list_input)
            
    #     else: raise(TypeError('Please pass the data as a 2-d array. Each object should be an n-dimensional vector. All objects should have the same dimension.'))

    #     prediction_input_len, prediction_input_dim = np.shape(self.prediction_input)
        
    #     if prediction_input_dim != self.data_dim:
    #         raise(AssertionError('The dimension of the prediction data does not match the dimension of the data used to train the SOM.'))

    #     unitary_covar_vector = [0.01] * prediction_input_dim

    #     prediction_results = np.full([prediction_input_len, self.labeling_data_dim - self.data_dim], np.nan)
    #     prediction_sigmas  = np.full([prediction_input_len, self.labeling_data_dim - self.data_dim], np.nan)

    #     for index in range(prediction_input_len):

    #         bmu_coords = find_bmu_coords(self.weights_map,
    #                                      self.prediction_input[index],
    #                                      self.prediction_stds[index])
    #         prediction_results[index] = np.sum(self.distribution_xs * self.labeled_map[*bmu_coords], axis = -1)

    #         # prediction_sigmas[index] = np.sqrt(np.sum(self.distribution_xs ** 2 * self.labeled_map[*bmu_coords], axis = -1) -\
    #         #                                    np.sum(self.distribution_xs * self.labeled_map[*bmu_coords], axis = -1))
        
    #     self.prediction_results = prediction_results
    #     # self.prediction_sigmas  = prediction_sigmas
    #     # return prediction_results

    # def predict(self,
    #             prediction_input,
    #             prediction_stds = None):
        
    #     self.prediction_stds = prediction_stds
        
    #     if len(np.shape(prediction_input)) == 2:
    #         self.prediction_input = np.array(prediction_input)

    #     elif (len(np.shape(prediction_input)) == 1) & (len(prediction_input) > 1):

    #         tuple_input = prediction_input.as_array()
    #         list_input = [list(values) for values in tuple_input]
    #         self.prediction_input = np.array(list_input)
            
    #     else: raise(TypeError('Please pass the data as a 2-d array. Each object should be an n-dimensional vector. All objects should have the same dimension.'))

    #     prediction_input_len, prediction_input_dim = np.shape(self.prediction_input)
        
    #     if prediction_input_dim != self.data_dim:
    #         raise(AssertionError('The dimension of the prediction data does not match the dimension of the data used to train the SOM.'))

    #     unitary_covar_vector = [0.01] * prediction_input_dim

    #     prediction_results = np.full([prediction_input_len, self.labeling_data_dim - self.data_dim], np.nan)
    #     prediction_sigmas  = np.full([prediction_input_len, self.labeling_data_dim - self.data_dim], np.nan)

    #     for index in range(prediction_input_len):

    #         bmu_coords = find_bmu_coords(self.weights_map,
    #                                      self.prediction_input[index],
    #                                      self.prediction_stds[index])
    #         prediction_results[index] = np.sum(self.distribution_xs * self.labeled_map[*bmu_coords], axis = -1)

    #         # prediction_sigmas[index] = np.sqrt(np.sum(self.distribution_xs ** 2 * self.labeled_map[*bmu_coords], axis = -1) -\
    #         #                                    np.sum(self.distribution_xs * self.labeled_map[*bmu_coords], axis = -1))
        
    #     self.prediction_results = prediction_results
    #     # self.prediction_sigmas  = prediction_sigmas
    #     # return prediction_results

    # def label_map(self):

    #     self.labeled_map = np.full([*self.mapsize, self.labeling_data_dim - self.data_dim],
    #                               np.nan)

    #     unitary_covar_vector = [1] * self.data_dim

    #     labeling_data_bmu_idx = {}
    #     for index in range(self.labeling_data_len):

    #         bmu_coords = find_bmu_coords(self.weights_map,
    #                                      self.labeling_data[index, :self.data_dim],
    #                                      unitary_covar_vector)
            
    #         if f'{bmu_coords}' in labeling_data_bmu_idx.keys():
    #             labeling_data_bmu_idx[f'{bmu_coords}'].append(index)
    #         else:
    #             labeling_data_bmu_idx[f'{bmu_coords}'] = []
    #             labeling_data_bmu_idx[f'{bmu_coords}'].append(index)

    #     self.labeling_data_bmu_idx = labeling_data_bmu_idx

    #     self.labeled_map = np.full([*self.mapsize, self.labeling_data_dim - self.data_dim],
    #                               np.nan)
        
    #     iteration_map = np.nditer(self.labeled_map[..., 0], flags = ['multi_index'])

    #     for _ in iteration_map:
            
    #         index = iteration_map.multi_index

    #         if f'{index}' in labeling_data_bmu_idx.keys():
    #             local_vectors = self.labeling_data[labeling_data_bmu_idx[f'{index}']]
    #             self.labeled_map[*index] = np.nanmedian(local_vectors[:, self.data_dim:], axis = 0)

    # def predict(self,
    #             prediction_input):
        
    #     if len(np.shape(prediction_input)) == 2:
    #         self.prediction_input = np.array(prediction_input)

    #     elif (len(np.shape(prediction_input)) == 1) & (len(prediction_input) > 1):

    #         tuple_input = prediction_input.as_array()
    #         list_input = [list(values) for values in tuple_input]
    #         self.prediction_input = np.array(list_input)
            
    #     else: raise(TypeError('Please pass the data as a 2-d array. Each object should be an n-dimensional vector. All objects should have the same dimension.'))

    #     prediction_input_len, prediction_input_dim = np.shape(self.prediction_input)
        
    #     if prediction_input_dim != self.data_dim:
    #         raise(AssertionError('The dimension of the prediction data does not match the dimension of the data used to train the SOM.'))

    #     unitary_covar_vector = [1] * prediction_input_dim

    #     prediction_results = np.full([prediction_input_len, self.labeling_data_dim - self.data_dim], np.nan)

    #     for index in range(prediction_input_len):

    #         bmu_coords = find_bmu_coords(self.weights_map,
    #                                      self.prediction_input[index],
    #                                      unitary_covar_vector)
    #         prediction_results[index] = self.labeled_map[*bmu_coords]
        
    #     self.prediction_results = prediction_results
    #     return prediction_results