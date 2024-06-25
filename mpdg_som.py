import numpy as np
from numpy import linalg as linalg

#assumptions: data is not covariant

class SelfOrganizingMap:

    def __init__(self,
                 mapsize,
                 dimension: int = None,
                 name: str = 'som',
                 initialization: str = 'random'
                 ):
        
        self.name = name
        self.use_covariance = False

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

    def load_standard_deviations(self,
                                 stds):
        
    #     if np.shape(variances) == np.shape(self.data):
    #         self.covar_matrix = 
    #     self.sigmas

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

    def build_SOM(self):

        SOM_space = self.mapsize.copy()
        SOM_space.append(self.data_dim)

        self.SOM = np.zeros(SOM_space)

    def chi_sq_dist(self,
                    weight_vector,
                    data_vector,
                    covar_vector = None):
        
        if ~self.use_covariance:
            covar_matrix = np.diagflat([1] * self.data_dim)
        
        elif (self.use_covariance) & (covar_vector == None):
            raise(ValueError('There is no covariance matrix for the given data vector!'))

        covar_matrix = np.diagflat(covar_vector)
        inv_covar_matrix = linalg.inv(covar_matrix)

        vector_difference = data_vector - weight_vector

        return np.dot(np.dot(vector_difference, inv_covar_matrix),vector_difference)
