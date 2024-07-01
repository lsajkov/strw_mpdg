# File meant to find optimal
# hyperparameters for SOM

from astropy.io import fits
from astropy.table import Table
import numpy as np

from mpdg_som import SelfOrganizingMap
import optuna

#load in data
cut_data_file = '/data2/lsajkov/mpdg/data_products/GAMA/GAMA_primtarg_snr100_lms6_12_25jun2024.fits'

with fits.open(cut_data_file) as cat:
    GAMA_vect_data = Table(cat[1].data)

GAMA_vect_data.add_column(GAMA_vect_data['r_mag_err'], index = 4, name = 'surf_bright_r_err')

#Select the needed data
input_data = GAMA_vect_data['r_mag', 'gr_color', 'surf_bright_r']
input_stds = GAMA_vect_data['r_mag_err', 'gr_color_err', 'surf_bright_r_err']

# input_labels = GAMA_vect_data['log_stellar_mass', 'redshift']

# tuple_labels = input_labels.as_array()
# list_labels = [list(values) for values in tuple_labels]
# input_labels = np.array(list_labels)

data_cut = 1000 #use up to this much of the data
randomized_idx = np.arange(0, len(input_data))
np.random.shuffle(randomized_idx)
randomized_idx = randomized_idx[:data_cut]

#set SOM metaparameters
name = 'mpdg_optuna'

initialization = 'pca'
termination = 'maximum_steps'
learning_rate_function = 'power_law'
neighborhood_function = 'gaussian'
error_estimator = 'quantization_error'

#initalize the SOM with Optuna hyperparameter search

#DEBUG
maximum_steps = 10
#DEBUG

def ObjectiveFunction(trial):

    mapsize_len   = trial.suggest_int('mapsize_len', 5, 30)
    mapsize_wid   = trial.suggest_int('mapsize_wid', 5, 30)
    learning_rate = trial.suggest_float('learning_rate', 0, 1)
    kernel_spread = trial.suggest_float('kernel_spread', 0.5, 5)
    # maximum_steps = trial.suggest_int('maximum_steps', 25, 150)

    SOM = SelfOrganizingMap(
        name                   = name,
        mapsize                = [mapsize_len, mapsize_wid],
        dimension              = 2,
        initialization         = initialization,
        termination            = termination,
        learning_rate_function = learning_rate_function,
        neighborhood_function  = neighborhood_function,
        error_estimator        = error_estimator,
        learning_rate          = learning_rate,
        kernel_spread          = kernel_spread,
        maximum_steps          = maximum_steps,
        error_thresh           = 1e-4)
    
    SOM.load_data(input_data[randomized_idx])
    SOM.normalize_data()

    SOM.load_standard_deviations(input_stds[randomized_idx])
    SOM.normalize_standard_deviations()

    SOM.build_SOM()

    error = SOM.train()

    return error

study = optuna.create_study(study_name = 'mpdg_som',
                            direction = 'minimize')
study.optimize(ObjectiveFunction, n_trials = 250)