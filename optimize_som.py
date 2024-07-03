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

data_cut = 18000 #use up to this much of the data (-1 uses entire dataset)
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
maximum_steps = 20

def ObjectiveFunction(trial):

    mapsize_len   = trial.suggest_int('mapsize_len', 20, 50)
    mapsize_wid   = trial.suggest_int('mapsize_wid', 20, 50)
    learning_rate = trial.suggest_float('learning_rate', 0.5, 1)
    kernel_spread = trial.suggest_float('kernel_spread', 0.5, 20)
    # maximum_steps = trial.suggest_int('maximum_steps', 5, 20)

    global SOM

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
        error_thresh           = 0.05)
    
    SOM.load_data(input_data[randomized_idx])
    SOM.normalize_data()

    SOM.load_standard_deviations(input_stds[randomized_idx])
    SOM.normalize_standard_deviations()

    SOM.build_SOM()

    error = SOM.train()

    return error

import os
from datetime import datetime

todays_date = datetime.today().strftime('%d%b%y')
study_name = f'SOM_optuna_{todays_date}'
n_trials = 25

start_time = datetime.today().strftime('%Hh%Mm')

log_file = f'/data2/lsajkov/mpdg/strw_mpdg/optimization_results/output_log_{todays_date}_{start_time}'
with open(log_file, 'w') as log:
    log.write(f'SOM Optuna optimization process. Started on {todays_date} at {start_time}.\n')
    log.write(f'trial\terror\tparams\n')

if not os.path.exists(f'/data2/lsajkov/mpdg/saved_soms/{todays_date}_{start_time}'):
    os.mkdir(f'/data2/lsajkov/mpdg/saved_soms/{todays_date}_{start_time}')


def dump_to_file(study, frozen_trial):

    ft = frozen_trial
    with open(log_file, 'a') as log:
        log.write(f'{ft.number}\t{ft.value:3f}\t{ft.params}\n')
    
    save_map_file = f'/data2/lsajkov/mpdg/saved_soms/{todays_date}_{start_time}/trial{ft.number}'
    np.save(save_map_file, SOM.weights_map)



study = optuna.create_study(study_name = study_name,
                            direction = 'minimize')
study.optimize(ObjectiveFunction,
               n_trials = n_trials,
               callbacks = [dump_to_file])

with open(log_file, 'a') as log:
    
    bft = study.best_trial
    log.write('best trial:\n')
    log.write(f'{bft.number}\t{bft.value:.3f}\t{bft.params}')

# #save outputs
# import json

# results_directory = f'/data2/lsajkov/mpdg/strw_mpdg/{study_name}'

# # json_file_contents = {}
# # for trial in study.get_trials():
# #     json_file_contents[f'trial{trial.number}'] = {}
#     json_file_contents[f'trial{trial.number}']['error'] = trial.values
#     for param in trial.params.keys():
#         json_file_contents[f'trial{trial.number}'][param] = trial.params[param]

# with open(f'{results_directory}/SOM_optuna_trials.json', 'w') as json_file:
#     json.dump(json_file_contents, json_file)


