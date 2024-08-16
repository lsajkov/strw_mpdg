from fitsio import FITS
import os
import numpy as np
import healpy as hp

hmap = hp.read_map('/data2/lsajkov/mpdg/data_products/KiDS_randoms_mask/kids_dr4.0_fp_try1.fits')
nside = int((len(hmap)/12)**0.5)

def check_in_kids(ra, dec):
    ipix = hp.ang2pix(nside, ra, dec, lonlat = 1)
    idx = hmap[ipix] == 1.0
    coords_in_KiDS = np.array(list(zip(ra[idx], dec[idx])),
                              dtype = [('ra', '<f4'), ('dec', '<f4')])
    return coords_in_KiDS

for catalog_number in range(8, 20):
    # catalog_number = sys.argv[1]
    print(f'Finding randoms in catalog {catalog_number}')

    randoms_catalog_path = f'/data2/lsajkov/mpdg/data/legacy_survey/allsky_randoms/randoms-allsky-1-{catalog_number}.fits'
    # randoms_catalog_path = '/data2/lsajkov/mpdg/data_products/KiDS/SOM/KiDS_SOM_panchrom_07Aug24.fits'
    randoms_catalog = FITS(randoms_catalog_path)[1]

    if os.path.exists(f'/data2/lsajkov/mpdg/data_products/KiDS_randoms/randoms_inKiDS_{catalog_number}.fits'):
        os.remove(f'/data2/lsajkov/mpdg/data_products/KiDS_randoms/randoms_inKiDS_{catalog_number}.fits')
    randoms_in_KiDS_catalog = FITS(f'/data2/lsajkov/mpdg/data_products/KiDS_randoms/randoms_in_KiDS_{catalog_number}.fits',
                                'rw')

    nrows = randoms_catalog.get_nrows()
    nsteps = 1000
    stepsize = nrows//nsteps

    randoms_in_KiDS_catalog.write(check_in_kids(randoms_catalog[0:stepsize]['RA'],
                                                randoms_catalog[0:stepsize]['DEC']))
    print(f'Completed 1/{nsteps}\t\t[{30 * ' '}]', end = '\r')

    for ii in range(1, nsteps):
        idx_lo = stepsize * ii
        idx_hi = stepsize * (ii + 1)
        if idx_hi > nrows: idx_hi = nrows - 1
        randoms_in_KiDS_catalog[1].append(check_in_kids(randoms_catalog[idx_lo:idx_hi]['RA'],
                                                        randoms_catalog[idx_lo:idx_hi]['DEC']))
        print(f'Completed {ii:5>}/{nsteps}\t\t[{30 * ii//nsteps * '='}{(30 - (30 * ii//nsteps)) * ' '}]', end = '\r')
    print(f'{100 * ' '}', end = '\r')