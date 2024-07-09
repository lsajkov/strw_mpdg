# File meant to find optimal
# hyperparameters for SOM

from astropy.io import fits
from astropy.table import Table
import numpy as np

from mpdg_som import SelfOrganizingMap
import optuna

#load in data
data_file = '/data2/lsajkov/mpdg/data_products/GAMA/GAMA_SOM_training_catalog_08Jul24.fits'

with fits.open(data_file) as cat:
    input_catalog_complete = Table(cat[1].data)

#Select the needed data
input_data = input_catalog_complete['gr_col', 'ug_col', 'ri_col']
input_stds = input_catalog_complete['gr_col_err', 'ug_col_err', 'ri_col_err']

input_labels = input_catalog_complete['gr_col', 'ug_col', 'ri_col', 'log_mstar', 'redshift']
input_label_stds = input_catalog_complete['gr_col_err', 'ug_col_err', 'ri_col_err']

data_cut = 30000 #int(len(input_data))#use up to this much of the data (-1 uses entire dataset)

#pick a random subset of the data, as to avoid sampling bias
randomized_idx = np.arange(0, len(input_data))
np.random.shuffle(randomized_idx)
randomized_data_idx = randomized_idx[:data_cut]
randomized_label_idx = randomized_idx[data_cut:]

#set SOM metaparameters
name = 'mpdg_optuna'

initialization         = 'pca'
termination            = 'either'
learning_rate_function = 'power_law'
neighborhood_function  = 'gaussian'
error_estimator        = 'quantization_error'

#initalize the SOM with Optuna hyperparameter search
maximum_steps = 10

def ObjectiveFunction(trial):

    mapsize_len   = trial.suggest_int('mapsize_len', 20, 60)
    mapsize_wid   = trial.suggest_int('mapsize_wid', 20, 60)
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
    
    SOM.load_data(input_data[randomized_data_idx])
    SOM.normalize_data()

    SOM.load_standard_deviations(input_stds[randomized_data_idx])
    SOM.normalize_standard_deviations()

    SOM.build_SOM()

    SOM.train()

    SOM.load_labeling_data(input_labels[randomized_label_idx],
                           parameter_names = ['log_mstar', 'redshift'])
    SOM.normalize_labeling_data()

    SOM.label_map()
    SOM.predict(SOM.labeling_data[:, :3])

    rms_error = np.sqrt(np.sum((SOM.labeling_data[:, 3:] - SOM.prediction_results) ** 2, axis = 0)/len(SOM.labeling_data))

    print(rms_error[0] + 5 * rms_error[1])
    return rms_error[0] + 5 * rms_error[1]

import os
from datetime import datetime

todays_date = datetime.today().strftime('%d%b%y')
start_time  = datetime.today().strftime('%Hh%Mm')
study_name  = f'SOM_optuna_{todays_date}'

n_trials = 25

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
                            direction  = 'minimize')
study.optimize(ObjectiveFunction,
               n_trials  = n_trials,
               callbacks = [dump_to_file])

with open(log_file, 'a') as log:
    
    bft = study.best_trial
    log.write('best trial:\n')
    log.write(f'{bft.number}\t{bft.value:.3f}\t{bft.params}\n')
