import numpy as np

class SelfOrganizingMap:

    def __init__(self,
                 mapsize,
                 name: str = 'som',
                 ):
        
        self.name = name

        if isinstance(mapsize, int):
            self.mapsize = [mapsize, mapsize]

        elif isinstance(mapsize, list) & (len(mapsize) == 2):
            self.mapsize = mapsize
        
        elif isinstance(mapsize, tuple) & (len(mapsize) == 2):
            self.mapsize = [mapsize[0], mapsize[1]]

        else: raise(ValueError('Mapsize must be an integer or a pair of numbers in list or tuple.'))

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
        self.dim  = np.shape(self.data)[1]

        if variable_names != None:
            self.variable_names = variable_names
        
        elif variable_names == None:
            self.variable_names = [f'var{i}' for i in range(self.dim)]

    def data_statistics(self):

        print('| Data statistics ')
        print('stat\t', end = '')
        for i in range(self.dim): print(self.variable_names[i], end = '\t')
        print('\nmin\t', end = '')
        for i in range(self.dim): print(f'{np.min(self.data[:, i]):.3f}', end = '\t')
        print('\nmax\t', end = '')
        for i in range(self.dim): print(f'{np.max(self.data[:, i]):.3f}', end = '\t')
        print('\nmean\t', end = '')
        for i in range(self.dim): print(f'{np.mean(self.data[:, i]):.3f}', end = '\t')
        print('\nmedian\t', end = '')
        for i in range(self.dim): print(f'{np.median(self.data[:, i]):.3f}', end = '\t')

    
