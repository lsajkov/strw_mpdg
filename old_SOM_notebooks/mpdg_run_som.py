#.py counterpart to mpdg_run_som.ipynb
#for remote access and large tasks

from astropy.io import fits
from astropy.table import Table

import numpy as np

#Load in training data
data_file = '/data2/lsajkov/mpdg/data_products/GAMA/GAMA_SOM_training_catalog_04Jul24.fits'

with fits.open(data_file) as cat:
    input_catalog_complete = Table(cat[1].data)

#Select the needed data
input_data = input_catalog_complete['gr_col', 'ug_col']
input_stds = input_catalog_complete['gr_col_err', 'ug_col_err']

print(f'Len of input data: {len(input_data)}')

from mpdg_som import SelfOrganizingMap

#Set parameters
name = 'mass_profile_dwarf_galaxies' #name of the SOM

mapsize   = [40, 40] #size of the map. pass as a list of dimensions OR as an integer (also pass number of dimensions)
dimension = None

initialization         = 'pca' #random or pca (principal component analysis)
termination            = 'either' #when to stop learning. maximum_steps = stop when maximum_steps have elapsed. error_thresh = stop when the error is below this threshold. either = stop when either condition is fulfilled
learning_rate_function = 'power_law' #which learning rate function to use. currently implemented: power_law
neighborhood_function  = 'gaussian' #which neighborhood function to use. currently implemented: gaussian
error_estimator        = 'quantization_error' #which error estimation function to use. currently implemented: max_misalignment

learning_rate = 0.7 #used to adjust the learning rate function
kernel_spread = 20 #used to adjust the neighborhood function
maximum_steps = 10 #used to adjust the learning rate and neighborhood functions
error_thresh  = 0.01 #used to stop the SOM if termination = 'error thresh'

#Declare the SOM
SOM = SelfOrganizingMap(
    name                   = name,
    mapsize                = mapsize,
    dimension              = dimension,
    initialization         = initialization,
    termination            = termination,
    learning_rate_function = learning_rate_function,
    neighborhood_function  = neighborhood_function,
    error_estimator        = error_estimator,
    learning_rate          = learning_rate,
    kernel_spread          = kernel_spread,
    maximum_steps          = maximum_steps,
    error_thresh           = error_thresh
)

data_cut = 50000 #use up to this much of the data (-1 for entire dataset)
randomized_idx = np.arange(0, len(input_data))
np.random.shuffle(randomized_idx)
randomized_idx = randomized_idx[:data_cut]

SOM.load_data(input_data[randomized_idx],
              variable_names = ['g-r', 'u-g'])
SOM.normalize_data()

SOM.load_standard_deviations(input_stds[randomized_idx])
SOM.normalize_standard_deviations()

SOM.data_statistics()

#Initialize the SOM
SOM.build_SOM()

#Visualize SOM before training
SOM.show_map(cmap = 'jet',
             save = True, save_path = '/data2/lsajkov/mpdg/figures/SOM_pretraining_08jul24')

#Look at initial quantization error
from mpdg_som_utils import SOM_ErrorEstimators

initial_quant_error = SOM_ErrorEstimators.quantization_error(SOM.weights_map,
                                                             SOM.data,
                                                             SOM.bmu_indices)
print(f'| Initial quantization error: {initial_quant_error:.3f}')

#Train the som
SOM.train()

#Visualize the SOM after training
SOM.show_map(cmap = 'jet',
             save = True, save_path = '/data2/lsajkov/mpdg/figures/SOM_posttraining_08jul24')

SOM.save_outputs('/data2/lsajkov/mpdg/strw_mpdg/optimization_results/GAMA_SOM_08jul24',
                 save_weights = True, save_parameters = True)