import os
import fnmatch

import numpy as np
import fitsio

from astropy.table import Table, vstack, hstack

local_kids_dir = '/data2/lsajkov/mpdg/data/KiDS_multiband'
test_cat = fitsio.read(f'{local_kids_dir}/{os.listdir(local_kids_dir)[0]}', ext = 1)
test_cat = Table(test_cat)

catalog_keys = test_cat.keys()
photometric_bands = fnmatch.filter(catalog_keys,
                                  'MAG_GAAP_[!10]*')
photometric_bands = [mag_gaap.split('_')[-1] for mag_gaap in photometric_bands]

KiDS_SOM_catalog = Table()

total_tiles = len(os.listdir(local_kids_dir)); count = 0

print(f'Tiles completed: 0/{total_tiles} [{30 * " "}]', end = '\r')
for kids_multiband_cat in fnmatch.filter(os.listdir(local_kids_dir),
                                         'KiDS_DR4.*_*ugriZYJHKs_cat.fits'):

    catData = fitsio.read(f'{local_kids_dir}/{kids_multiband_cat}', ext = 1)
    catData = Table(catData)

    #define columns
    ID = catData['ID']

    ra  = catData['RAJ2000']
    dec = catData['DECJ2000']

    r_mag     = catData['MAG_AUTO']
    r_mag_err = catData['MAGERR_AUTO']

    gaap_magnitudes = Table()

    for band in photometric_bands:

        gaap_magnitudes.add_column(catData[f'MAG_GAAP_{band}'],
                                   name = f'{band}_gaap_mag')
        gaap_magnitudes.add_column(catData[f'MAGERR_GAAP_{band}'],
                                   name = f'{band}_mag_gaap_err')

    colors = Table()
    color_mask = np.ones(len(catData), dtype = bool)
    colors_lo = -1
    colors_hi = 3

    for i, band_hi in enumerate(photometric_bands):
        for band_lo in photometric_bands[i + 1:]:
            color_column = catData[f'MAG_GAAP_{band_hi}'] - catData[f'MAG_GAAP_{band_lo}']
            color_err_column = np.sqrt(catData[f'MAGERR_GAAP_{band_hi}'] ** 2 \
                                     + catData[f'MAGERR_GAAP_{band_lo}'] ** 2)

            colors.add_column(color_column,
                              name = f'{band_hi}{band_lo}_col')
            
            colors.add_column(color_err_column,
                              name = f'{band_hi}{band_lo}_col_err')

            color_mask = color_mask &\
                        (color_column > colors_lo) & (color_column < colors_hi) &\
                        ~np.isnan(color_column)

    redshift = catData['Z_B']
    redshift_err = (catData['Z_B_MAX'] - catData['Z_B_MIN'])/2
 
    flux_radius = catData['FLUX_RADIUS'] * 0.213


    #define masks
    # KiDS_flags_mask = catData['FLAG_GAAP_r'] == 0

    KiDS_MASK_mask         = ~((catData['MASK'] & 28668) > 0)
    KiDS_IMAFLAGS_ISO_mask = (catData['IMAFLAGS_ISO'] == 0)
    KiDS_CLASS_STAR_mask   = catData['CLASS_STAR'] < 0.5
    KiDS_SG2DPHOT_mask     = catData['SG2DPHOT'] == 0
    KiDS_SGFLAG_mask       = catData['SG_FLAG'] == 1
    
    SNR_thresh = 5
    SNR_mask = catData['FLUX_GAAP_r']/catData['FLUXERR_GAAP_r'] > SNR_thresh
    # SNR_mask = np.ones(len(catData), dtype = bool)

    KiDS_flags_mask = np.ones(len(catData), dtype = bool)
    
    for band in photometric_bands:

        # band_SNR = catData[f'FLUX_GAAP_{band}']/catData[f'FLUXERR_GAAP_{band}']
        # SNR_mask = SNR_mask & (band_SNR > SNR_thresh)

        KiDS_flags_mask = KiDS_flags_mask & (catData[f'FLAG_GAAP_{band}'] == 0)

    redshift_mask = catData['Z_B'] < 1

    r_mag_mask = r_mag > 18

    complete_mask = KiDS_MASK_mask &\
                    KiDS_IMAFLAGS_ISO_mask &\
                    KiDS_CLASS_STAR_mask &\
                    KiDS_SG2DPHOT_mask &\
                    KiDS_SGFLAG_mask &\
                    color_mask &\
                    r_mag_mask &\
                    SNR_mask &\
                    redshift_mask
    
    tile_KiDS_data = Table([ID,
                            ra, dec,
                            r_mag, r_mag_err,
                            flux_radius,
                            redshift, redshift_err],
                    names = ['ID',
                             'ra', 'dec',
                             'r_mag', 'r_mag_err',
                             'half_light_radius',
                             'redshift', 'redshift_err'])
    
    tile_KiDS_data.add_columns(colors.columns,
                               indexes = [5] * len(colors.colnames))
        
    tile_KiDS_data.add_columns(gaap_magnitudes.columns,
                               indexes = [5] * len(gaap_magnitudes.colnames))
    
    tile_KiDS_data = tile_KiDS_data[complete_mask]

    KiDS_SOM_catalog = vstack([KiDS_SOM_catalog,
                               tile_KiDS_data])

    count += 1
    frac = int(30 * count/total_tiles)
    print(f'Tiles completed: {count}/{total_tiles} [{frac * "*"}{(30 - frac) * " "}]', end = '\r')

KiDS_SOM_catalog.write('/data2/lsajkov/mpdg/data_products/KiDS/KiDS_SOM_panchrom_30Jul24.fits',
                       overwrite = False)