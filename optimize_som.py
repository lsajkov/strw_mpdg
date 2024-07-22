# File meant to find optimal
# hyperparameters for SOM

from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

from mpdg_som import SelfOrganizingMap
import optuna

#load in data
data_file = '/data2/lsajkov/mpdg/data_products/KiDS/KiDS_SOM_catalog_18Jul24.fits'
label_file = '/data2/lsajkov/mpdg/data_products/GAMA/GAMA_SOM_training_catalog_17Jul24.fits'

with fits.open(data_file) as cat:
    input_catalog_complete = Table(cat[1].data)

with fits.open(label_file) as cat:
    label_catalog_complete = Table(cat[1].data)

#Select the needed data
input_redshift_cut = input_catalog_complete['redshift'] <= 0.4
input_mag_cut      = input_catalog_complete['r_mag'] < 20.5
input_size_cut     = input_catalog_complete['half_light_radius'] < 5
input_total_cut    = input_redshift_cut & input_mag_cut & input_size_cut

label_redshift_cut = label_catalog_complete['redshift'] <= 0.4
label_mag_cut      = label_catalog_complete['r_mag'] < 20.5
label_size_cut     = label_catalog_complete['half_light_radius'] < 5
label_total_cut    = label_redshift_cut & label_size_cut & label_redshift_cut

input_data = input_catalog_complete[input_total_cut]['r_mag', 'gr_col', 'ug_col', 'ri_col', 'half_light_radius']
input_stds = input_catalog_complete[input_total_cut]['r_mag_err', 'gr_col_err', 'ug_col_err', 'ri_col_err']
input_stds.add_column([0.1] * len(input_catalog_complete[input_total_cut]), name = 'half_light_radius_err')

input_labels     = label_catalog_complete[label_total_cut]['r_mag','gr_col', 'ug_col', 'ri_col', 'half_light_radius', 'mstar', 'redshift']
input_label_stds = label_catalog_complete[label_total_cut]['r_mag_err', 'gr_col_err', 'ug_col_err', 'ri_col_err']
input_label_stds.add_column([0.1] * len(label_catalog_complete[label_total_cut]), name = 'half_light_radius_err')
input_label_stds.add_column([1] * len(label_catalog_complete[label_total_cut]), name = 'mstar_err_placeholder')
input_label_stds.add_column([1e-4] * len(label_catalog_complete[label_total_cut]), name = 'redshift_err_placeholder')

# tuple_labels = input_labels.as_array()
# list_labels  = [list(values) for values in tuple_labels]
# input_labels = np.array(list_labels)

data_cut = -1 #use up to this much of the data (-1 uses entire dataset)

#pick a random subset of the data, as to avoid sampling bias
randomized_idx = np.arange(0, len(input_data))
np.random.shuffle(randomized_idx)
randomized_data_idx  = randomized_idx[:data_cut]
randomized_label_idx = randomized_idx[data_cut:]

#set SOM metaparameters
name = 'mpdg_optuna'

initialization         = 'pca'
termination            = 'either'
learning_rate_function = 'power_law'
neighborhood_function  = 'gaussian'
error_estimator        = 'quantization_error'

#initalize the SOM with Optuna hyperparameter search
maximum_steps = 20

def ObjectiveFunction(trial):

    mapsize_len   = trial.suggest_int('mapsize_len', 20, 60)
    mapsize_wid   = trial.suggest_int('mapsize_wid', 20, 60)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.99)
    kernel_spread = trial.suggest_float('kernel_spread', 0.5, 20)
    # maximum_steps = trial.suggest_int('maximum_steps', 5, 20)

    global SOM

    SOM = SelfOrganizingMap(
        name                   = name,
        mapsize                = [mapsize_len, mapsize_wid],
        dimension              = None,
        initialization         = initialization,
        termination            = termination,
        learning_rate_function = learning_rate_function,
        neighborhood_function  = neighborhood_function,
        error_estimator        = error_estimator,
        learning_rate          = learning_rate,
        kernel_spread          = kernel_spread,
        maximum_steps          = maximum_steps,
        error_thresh           = 0.01)
    
    SOM.load_data(input_data,#[randomized_data_idx],
                  variable_names = ['r_mag', 'g-r', 'u-g', 'r-i', 'rad_50'])
    # SOM.normalize_data()

    SOM.load_standard_deviations(input_stds)#[randomized_data_idx])
    # SOM.normalize_standard_deviations()

    SOM.build_SOM()

    SOM.train()

    SOM.load_labeling_data(input_labels,#[randomized_label_idx],
                           parameter_names = ['mstar', 'redshift'])
    # SOM.normalize_labeling_data()

    SOM.load_labeling_standard_deviations(input_label_stds)#[randomized_label_idx])
    # SOM.normalize_labeling_standard_deviations()

    SOM.label_map()
    
    SOM.predict(SOM.labeling_data[:, :SOM.data_dim],
                SOM.label_variances[:, :SOM.data_dim])

    rms_error = np.sqrt(np.nansum((np.log10(SOM.labeling_data[:, SOM.data_dim]) -\
                                   np.log10(SOM.prediction_results[:, 0]))**2)/len(SOM.labeling_data))
    
    if np.isnan(rms_error):
        rms_error = 99
    
    return rms_error

import os
from datetime import datetime

todays_date = datetime.today().strftime('%d%b%y')
start_time  = datetime.today().strftime('%Hh%Mm')
study_name  = f'SOM_optuna_{todays_date}'

n_trials = 50

log_file = f'/data2/lsajkov/mpdg/strw_mpdg/optimization_results/output_log_{todays_date}_{start_time}'
with open(log_file, 'w') as log:
    log.write(f'SOM Optuna optimization process. Started on {todays_date} at {start_time}.\n')
    log.write(f'trial\terror\tparams\n')

if not os.path.exists(f'/data2/lsajkov/mpdg/saved_soms/{todays_date}_{start_time}'):
    os.mkdir(f'/data2/lsajkov/mpdg/saved_soms/{todays_date}_{start_time}')

def create_trial_directory(study, frozen_trial):

    ft = frozen_trial
    directory_path = f'/data2/lsajkov/mpdg/saved_soms/{todays_date}_{start_time}/trial{ft.number}'

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

def write_to_log(study, frozen_trial):

    ft = frozen_trial
    with open(log_file, 'a') as log:
        log.write(f'{ft.number}\t{ft.value:3f}\t{ft.params}\n')

def save_map(study, frozen_trial):

    ft = frozen_trial

    save_map_path = f'/data2/lsajkov/mpdg/saved_soms/{todays_date}_{start_time}/trial{ft.number}/'
    np.save(f'{save_map_path}/weights', SOM.weights_map, allow_pickle = True)
    np.save(f'{save_map_path}/prediction_results', SOM.prediction_results, allow_pickle = True)
    np.save(f'{save_map_path}/labeling_data', SOM.labeling_data, allow_pickle = True)

    SOM.show_map(cmap = 'jet',
                 save = True, save_path = f'{save_map_path}/trained_SOM')
    
    SOM.show_map(show_labeled = True,
                 cmap = 'jet',
                 save = True, save_path = f'{save_map_path}/labeled_SOM',
                 log_norm = ['mstar'])

    SOM.produce_occupancy_map(show_map = True, save_map = True, save_fig = True,
                              save_path = save_map_path,
                              vmin = 0, vmax = 250)
    
    SOM.produce_quality_map(show_map = True, save_map = True, save_fig = True,
                            save_path = save_map_path)

def save_plots(study, frozen_trial):

    ft = frozen_trial
    plot_path = f'/data2/lsajkov/mpdg/saved_soms/{todays_date}_{start_time}/trial{ft.number}/comparison_plot.png'

    fig = plt.figure(figsize = (26, 12))
    ax1 = fig.add_subplot(121)
    hxb1 = ax1.hexbin(np.log10(SOM.labeling_data[:, SOM.data_dim]),
                      np.log10(SOM.prediction_results[:, 0]),
                      mincnt = 1, cmap = 'jet',
                      vmin = 0, vmax = 250)
    
    MAD_mstar = np.nansum(np.abs(np.log10(SOM.labeling_data[:, SOM.data_dim ])\
                         - np.log10(SOM.prediction_results[:, 0])))/(len(SOM.prediction_results))

    ax1.axline([10, 10], slope = 1, color = 'red')

    ax1.set_xlim(7, 12)
    ax1.set_ylim(7, 12)
    ax1.set_xticks(np.arange(7, 13, 1))

    ax1.text(0.65, 0.05,
        f'MAD: {MAD_mstar:.3f} dex',
        transform = ax1.transAxes)

    ax1.set_xlabel('GAMA log$_{10} (M_*/M_{\odot})$\nTrue')
    ax1.set_ylabel('Predicted\nSOM log$_{10} (M_*/M_{\odot})$')

    fig.colorbar(ax = ax1, mappable = hxb1,
                 location = 'top', pad = 0.01,
                 label = '$N_{\mathrm{galaxies}}$')
    
    ax2 = fig.add_subplot(122)

    hxb2 = ax2.hexbin(SOM.labeling_data[:, SOM.data_dim + 1],
                      SOM.prediction_results[:, 1],
                      mincnt = 1, cmap = 'jet',
                      vmin = 0, vmax = 250)
     
    MAD_redsh = np.nansum(np.abs(np.log10(SOM.labeling_data[:, SOM.data_dim + 1])\
                         - np.log10(SOM.prediction_results[:, 1])))/(len(SOM.prediction_results))
       
    ax2.axline([0, 0], slope = 1, color = 'red')

    ax2.set_xlabel('GAMA redshift\nTrue')
    ax2.set_ylabel('Predicted\nSOM redshift')

    ax2.text(0.65, 0.05,
        f'MAD: {MAD_redsh:.3f} dex',
        transform = ax2.transAxes)
    
    ax2.set_xlim(0, 0.405)
    ax2.set_ylim(0, 0.405)
    
    fig.colorbar(ax = ax2, mappable = hxb2,
                 location = 'top', pad = 0.01,
                 label = '$N_{\mathrm{galaxies}}$')
    
    fig.savefig(plot_path, bbox_inches = 'tight')

study = optuna.create_study(study_name = study_name,
                            direction  = 'minimize')
study.optimize(ObjectiveFunction,
               n_trials  = n_trials,
               callbacks = [create_trial_directory,
                            write_to_log,
                            save_map,
                            save_plots])

with open(log_file, 'a') as log:
    
    bft = study.best_trial
    log.write('best trial:\n')
    log.write(f'{bft.number}\t{bft.value:.3f}\t{bft.params}\n')
