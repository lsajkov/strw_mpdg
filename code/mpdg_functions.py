#mpdg_functions.py
#File meant to collect all functions relevant to mpdg notebooks
#For now, only contains the load_KiDS_data function, which loads magnitude- and redshift-selected sources from an individual KiDS catalog

from fitsio import FITS

def load_KiDS_data(path):

    # cat_section = fitsio.read(path, ext = 1)
    cat_section = FITS(path)[1][:]
    
    mag_cut      = cat_section['r_mag'] < 20.5
    redshift_cut = cat_section['redshift'] < 0.4

    cat_section_cut = cat_section[redshift_cut & mag_cut]

    colnames = cat_section.dtype.names
    columns = ['r_mag', 'r_mag_err']
    columns +=[key for key in colnames\
               if key.endswith('col')]
    columns +=[key for key in colnames\
            if key.endswith('col_err')]
    columns +=['half_light_radius']
    columns +=['redshift']

    cat_section_selected = cat_section_cut[columns]
    
    coordinate_columns = ['ID', 'KiDS_tile', 'ra', 'dec']
    
    cat_section_coordinates = cat_section_cut[coordinate_columns]

    return cat_section_selected, cat_section_coordinates