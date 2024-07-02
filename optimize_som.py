# File meant to find optimal
# hyperparameters for SOM

from astropy.io import fits
from astropy.table import Table
import numpy as np
import datetime

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

data_cut = 1200 #use up to this much of the data (-1 uses entire dataset)
randomized_idx = np.arange(0, len(input_data))
np.random.shuffle(randomized_idx)
randomized_idx = randomized_idx[:data_cut]

#set SOM metaparameters
name = 'mpdg_optuna'

initialization = 'pca'
termination = 'either'
learning_rate_function = 'power_law'
neighborhood_function = 'gaussian'
error_estimator = 'quantization_error'

#initalize the SOM with Optuna hyperparameter search

def ObjectiveFunction(trial):

    mapsize_len   = trial.suggest_int('mapsize_len', 10, 30)
    mapsize_wid   = trial.suggest_int('mapsize_wid', 10, 30)
    learning_rate = trial.suggest_float('learning_rate', 0.5, 1)
    kernel_spread = trial.suggest_float('kernel_spread', 0.5, 15)
    maximum_steps = trial.suggest_int('maximum_steps', 5, 20)

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
        error_thresh           = 0.1)
    
    SOM.load_data(input_data[randomized_idx])
    SOM.normalize_data()

    SOM.load_standard_deviations(input_stds[randomized_idx])
    SOM.normalize_standard_deviations()

    SOM.build_SOM()

    error = SOM.train()

    return error

from datetime import datetime
todays_date = datetime.today().strftime('%d%b%y')
study_name = f'SOM_optuna_{todays_date}'
n_trials = 25

study = optuna.create_study(study_name = study_name,
                            direction = 'minimize')
study.optimize(ObjectiveFunction, n_trials = n_trials)

#save outputs
import json

results_directory = f'/data2/lsajkov/mpdg/strw_mpdg/{study_name}'

json_file_contents = {}
for trial in study.get_trials():
    json_file_contents[f'trial{trial.number}'] = {}
    json_file_contents[f'trial{trial.number}']['error'] = trial.values
    for param in trial.params.keys():
        json_file_contents[f'trial{trial.number}'][param] = trial.params[param]

with open(f'{results_directory}/SOM_optuna_trials.json', 'w') as json_file:
    json.dump(json_file_contents, json_file)

