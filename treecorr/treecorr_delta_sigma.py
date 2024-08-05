# Measure excess surface density for a lens bin/source bin pair.
import sys
from glob import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import fitsio

from astropy.io import fits, ascii
from astropy.table import Table
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0 = 100, Om0 = 0.3, Ode0 = 0.7)


import treecorr

""" KiDS SOM-derived lens tomographic bins.
  | Bin | <z>  | <Mstar> |
  | 1   | 0.07 |  8.58   |
  | 2   | 0.09 |  9.20   |
  | 3   | 0.16 |  9.92   | NOT used in this analysis

"""

""" KiDS-1000 source tomographic bins. Drawn from Giblin et al. (2020):
  | Bin |   z_B range   |        m        |
  | 1   |   0.1<z<0.3   | -0.009 pm 0.019 |
  | 2   |   0.3<z<0.5   | -0.011 pm 0.020 |
  | 3   |   0.5<z<0.7   | -0.015 pm 0.017 |
  | 4   |   0.7<z<0.9   |  0.002 pm 0.012 |
  | 5   |   0.9<z<1.2   |  0.007 pm 0.010 |
"""

correction_m_array = [-0.009, -0.011, -0.015, 0.002, 0.007]

def degree_to_hMpc(degree, redshift):
    radian = (degree * u.degree).to(u.radian)
    angular_dist_Mpc = cosmo.angular_diameter_distance(redshift).value
    hMpc = angular_dist_Mpc * radian
    return hMpc

def hMpc_to_degree(hMpc, redshift):
    angular_dist_Mpc = cosmo.angular_diameter_distance(redshift).value
    radian = hMpc/angular_dist_Mpc
    degree = (radian * u.radian).to(u.degree)
    return degree.value

######
###  Import & define catalogs

lens_bin_directory = '/data2/lsajkov/mpdg/data_products/predicted_catalogs/KiDS_dwarf_galaxy_candidates'
srce_bin_directory = '/data2/lsajkov/mpdg/data/KiDS/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat'

lens_n_z_directory = '/data2/lsajkov/mpdg/data_products/KiDS/WL/n_z'
srce_n_z_directory = '/data2/lsajkov/mpdg/data/KiDS/SOM_N_of_Z'

lens_bin = sys.argv[1]
srce_bin = sys.argv[2]

# Lens catalog
lens_catalog_path = f'{lens_bin_directory}/KiDS_dwarf_galaxy_bin{lens_bin}.fits'

lens_ra  = fitsio.read(lens_catalog_path, columns = ['ra'])
lens_dec = fitsio.read(lens_catalog_path, columns = ['dec'])

lens_catalog = treecorr.Catalog(ra = lens_ra, dec = lens_dec,
                                ra_units = 'degrees', dec_units = 'degrees')
print('Loaded lens catalog')

lens_n_z_path = f'{lens_n_z_directory}/KiDS_n_z_bin{lens_bin}'
lens_n_z_array = ascii.read(lens_n_z_path)

# Source catalog
srce_catalog_path = f'{srce_bin_directory}/KiDS_DR4.1_WL_bin{srce_bin}.fits'

srce_ra      = fitsio.read(srce_catalog_path, columns = ['RAJ2000'],  ext = 1)
srce_dec     = fitsio.read(srce_catalog_path, columns = ['DECJ2000'], ext = 1)
srce_g1      = fitsio.read(srce_catalog_path, columns = ['e1'],       ext = 1)
srce_g2      = fitsio.read(srce_catalog_path, columns = ['e2'],       ext = 1)
srce_weights = fitsio.read(srce_catalog_path, columns = ['weight'],   ext = 1)

srce_catalog = treecorr.Catalog(ra = srce_ra, dec = srce_dec,
                                ra_units = 'degrees', dec_units = 'degrees',
                                g1 = srce_g1, g2 = srce_g2,
                                w  = srce_weights)
print('Loaded source catalog')

srce_n_z_path = f'{srce_n_z_directory}/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{srce_bin}_Nz.asc'
srce_n_z_array = ascii.read(srce_n_z_path)

######
### Define bins
lens_n_z_expected = np.dot(lens_n_z_array['col0'], lens_n_z_array['col1'])

log10_hMpc_bin_lo = -2
log10_hMpc_bin_hi = 1
bins = int(sys.argv[3])
hMpc_bins = np.logspace(log10_hMpc_bin_lo, log10_hMpc_bin_hi, bins)

degree_bins = hMpc_to_degree(hMpc_bins, lens_n_z_expected)

######
### Define treecorr config

config = {'nbins':     bins,
          'min_sep':   degree_bins[0],
          'max_sep':   degree_bins[-1],
          'sep_units': 'degree'}

######
### Calculate shear correlation

ngc = treecorr.NGCorrelation(config = config)
ngc.process(lens_catalog, srce_catalog)

######
### Calculate average surface critical density

G_in_pc_msun_s = const.G.to(u.parsec**3/(u.Msun * u.s**2))
c_in_pc_s      = const.c.to(u.parsec/u.s)

constant_factor = 4 * np.pi * G_in_pc_msun_s / (c_in_pc_s**2)

#lens integral
lens_redshifts     = lens_n_z_array['col0']
lens_dz            = lens_redshifts[1] - lens_redshifts[0] #the redshift bins are linear, all deltas are the same
lens_ang_diam_dist = cosmo.angular_diameter_distance(lens_redshifts).to(u.parsec)
lens_n_z           = lens_n_z_array['col1']
lens_n_z          /= np.sum(lens_n_z * lens_dz) #normalize n(z) so it integrates to 1
print(f'Lens n(z) integrates to {np.sum(lens_n_z * lens_dz):.2f}')

lens_integral = np.sum(lens_n_z * (1 + lens_redshifts)**2 * lens_ang_diam_dist * lens_dz)

#source integral
behind_lens_idx = srce_n_z_array['col1'] > lens_n_z_expected

srce_redshifts = srce_n_z_array[behind_lens_idx]['col1']
srce_dz        = srce_redshifts[1] - srce_redshifts[0] #the redshift bins are linear, all deltas are the same
srce_ang_diam_dist = cosmo.angular_diameter_distance(srce_redshifts).to(u.parsec)
srce_ang_diam_dist_w_lens = cosmo.angular_diameter_distance_z1z2(lens_n_z_expected, srce_redshifts).to(u.parsec)
srce_n_z = srce_n_z_array[behind_lens_idx]['col2']
srce_n_z /= np.sum(srce_n_z * lens_dz)
print(f'Source n(z) integrates to {np.sum(srce_n_z * lens_dz):.2f}')

srce_integral = np.sum(srce_n_z * srce_ang_diam_dist_w_lens / srce_ang_diam_dist * srce_dz)

#total integral
avg_sigma_crit = constant_factor * lens_integral * srce_integral
print(f'Units of average critical density: {avg_sigma_crit.unit}')

######
### Get excess surface density

shear_correlation_real = ngc.xi
shear_correlation_imag = ngc.xi_im

shear_correlation_covar = ngc.varxi

correction_m = correction_m_array[int(srce_bin) - 1]

excess_surf_density = shear_correlation_real/(1 + correction_m)/avg_sigma_crit
excess_surf_density_covar = shear_correlation_covar/(1 + correction_m)/avg_sigma_crit

save_results_directory = '/data2/lsajkov/mpdg/data_products/WL_excess_surf_density_results/05Aug24'

final_results = Table([np.round(degree_bins, 3),
                       np.round(hMpc_bins, 3),
                       np.round(shear_correlation_real, 6),
                       np.round(shear_correlation_imag, 6),
                       np.round(shear_correlation_covar, 6),
                       np.round(excess_surf_density.value, 6),  
                       np.round(excess_surf_density_covar.value, 6),
                       np.round([avg_sigma_crit.value] * len(degree_bins), 3)],
               names = ['R[degrees]',
                        'R[h-1 Mpc]',
                        'gamma_T_real',
                        'gamma_T_imag',
                        'gamma_T_covar',
                        'deltaSigma[h Msun pc-2]',
                        'covar_deltaSigma[h Msun pc-2]',
                        'averageSigmaCrit[pc2 Msun-1]'])

ascii.write(final_results, f'{save_results_directory}/output_lensbin{lens_bin}_srcebin{srce_bin}.dat',
            overwrite = True)

print(f'Final unit of excess surface density: {excess_surf_density.unit}')

