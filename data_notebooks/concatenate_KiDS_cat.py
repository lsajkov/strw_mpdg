from fitsio import FITS

KiDS_directory = '/data2/lsajkov/mpdg/data_products/KiDS/SOM/KiDS_SOM_panchrom_07Aug24'
KiDS_catalog_names = ['KiDS_SOM_panchrom_07Aug24_sec1.fits',
                      'KiDS_SOM_panchrom_07Aug24_sec2.fits',
                      'KiDS_SOM_panchrom_07Aug24_sec3.fits',
                      'KiDS_SOM_panchrom_07Aug24_sec4.fits',
                      'KiDS_SOM_panchrom_07Aug24_sec5.fits',
                      'KiDS_SOM_panchrom_07Aug24_sec6.fits',
                      'KiDS_SOM_panchrom_07Aug24_sec7.fits',
                      'KiDS_SOM_panchrom_07Aug24_sec8.fits',
                      'KiDS_SOM_panchrom_07Aug24_sec9.fits',
                      'KiDS_SOM_panchrom_07Aug24_sec10.fits',
                      'KiDS_SOM_panchrom_07Aug24_sec11.fits']
new_catalog_name = '/data2/lsajkov/mpdg/data_products/KiDS/SOM/KiDS_SOM_panchrom_07Aug24.fits'

full_KiDS_catalog = FITS(new_catalog_name, 'rw')

first_cat = True
count = 0
print(f'Completed {count}/{len(KiDS_catalog_names)}\t[{30 * ' '}]')
for catalog in KiDS_catalog_names:
    add_KiDS_catalog = FITS(f'{KiDS_directory}/{catalog}')[1].read()
    if count == 0:
        full_KiDS_catalog.write(add_KiDS_catalog)
    else:
        full_KiDS_catalog[1].append(add_KiDS_catalog)
    count += 1
    print(f'Completed {count}/{len(KiDS_catalog_names)}\t[{30 * count//len(KiDS_catalog_names) * '='}{(30 - (30 * count//len(KiDS_catalog_names))) * ' '}]')

# full_KiDS_catalog.close()