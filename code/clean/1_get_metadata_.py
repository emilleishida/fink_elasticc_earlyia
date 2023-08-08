import pandas as pd
import numpy as np
import glob

from copy import deepcopy
import os

def extract_history(history_list: list, field: str) -> list:
    """Extract the historical measurements contained in the alerts
    for the parameter `field`.

    Parameters
    ----------
    history_list: list of dict
        List of dictionary from alert[history].
    field: str
        The field name for which you want to extract the data. It must be
        a key of elements of history_list
    
    Returns
    ----------
    measurement: list
        List of all the `field` measurements contained in the alerts.
    """
    if history_list is None:
        return []
    try:
        measurement = [obs[field] for obs in history_list]
    except KeyError:
        print('{} not in history data'.format(field))
        measurement = []

    return measurement

def extract_field(alert: dict, category: str, field: str) -> np.array:
    """ Concatenate current and historical observation data for a given field.
    
    Parameters
    ----------
    alert: dict
        Dictionnary containing alert data
    category: str
        prvDiaSources or prvDiaForcedSources
    field: str
        Name of the field to extract.
    
    Returns
    ----------
    data: np.array
        List containing previous measurements and current measurement at the
        end. If `field` is not in the category, data will be
        [alert['diaSource'][field]].
    """
    data = np.concatenate(
        [
            [alert["diaSource"][field]],
            extract_history(alert[category], field)
        ]
    )
    return data

taxonomy = {111: 'Ia',
           112: 'Ibc',
           113: 'II',
           114: 'Iax',
           115: '91bg',
           120: 'FastOther',
           121: 'KN',
           122: 'Mdwarf',
           123: 'DwarfNovae',
           124: 'mLens',
           130: 'LongOther',
           131: 'SLSN',
           132: 'TDE',
           133: 'ILOT',
           134: 'CART',
           135: 'PISN',
           210: 'PeriodicOther',
           211: 'Cepheid',
           212: 'RRLyrae',
           213: 'dScuti',
           214: 'EB',
           215: 'LPVMira',
           220: 'NonPeriodicOther',
           221: 'AGN'}

taxonomy_inv = {}
for key in taxonomy.keys():
    taxonomy_inv[taxonomy[key]] = key

#*******************************************************************
bands = ['u', 'g', 'r', 'i', 'z', 'Y']

dir_output = '/media/ELAsTICC/Fink/first_year/with_metadata/'

cols = ['diaObjectId', 'alertId', 'classId', 'cmidPointTai', 'cpsFlux', 'cpsFluxErr', 'cfilterName', 'hostgal_sep', 'hostgal_zphot', 
        'hostgal_ellipticity', 'hostgal_sqradius', 'hostgal_zspec', 'hostgal_zspec_err', 'mwebv', 
        'mwebv_err', 'z_final', 'z_final_err', 'nobs', 'snr'] + \
       ['hostgal_mag_' + item for item in bands] + ['hostgal_magerr_' + item for item in bands] + \
       ['hostgal_zphot_q' + str(step).zfill(3) for step in range(0,101,10)] + ['delta_t', 'rising']

class_name = 'Ia'
class_number = taxonomy_inv[class_name]


raw_data_dir = '/media/ELAsTICC/Fink/first_year/ftransfer_elasticc_2023-02-15_946675/'
#*******************************************************************

if not os.path.isdir(dir_output + 'classId=' + str(class_number) + '/'):
    os.makedirs(dir_output + 'classId=' + str(class_number) + '/')

# read one specific class
flist_raw = glob.glob(raw_data_dir + 'classId=' + str(class_number) + '/part-*.parquet')



for fname in flist_raw:
    pdf = pd.read_parquet(fname)
    pdf['classId'] = class_number

    # extract light curve and metadata
    pdf['diaObjectId'] = [pdf['diaObject'][i]['diaObjectId'] for i in range(pdf.shape[0])]
    pdf['cpsFlux'] = pdf[['diaSource', 'prvDiaForcedSources']]\
                    .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'psFlux'), axis=1)
    pdf['cpsFluxErr'] = pdf[['diaSource', 'prvDiaForcedSources']]\
                    .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'psFluxErr'), axis=1)
    pdf['cfilterName'] = pdf[['diaSource', 'prvDiaForcedSources']]\
                    .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'filterName'), axis=1)
    pdf['cmidPointTai'] = pdf[['diaSource', 'prvDiaForcedSources']]\
                    .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'midPointTai'), axis=1) 
    pdf['nobs'] = [pdf['diaSource'].iloc[i]['nobs'] for i in range(pdf.shape[0])]
    pdf['snr'] = [pdf['diaSource'].iloc[i]['snr'] for i in range(pdf.shape[0])]
    pdf['hostgal_snsep'] = [pdf['diaObject'].iloc[i]['hostgal_snsep'] for i in range(pdf.shape[0])]
    pdf['hostgal_zphot'] = [pdf['diaObject'].iloc[i]['hostgal_zphot'] for i in range(pdf.shape[0])]
    pdf['hostgal_zphot_err'] = [pdf['diaObject'].iloc[i]['hostgal_zphot_err'] for i in range(pdf.shape[0])]
    pdf['hostgal_ellipticity'] = [pdf['diaObject'].iloc[i]['hostgal_ellipticity'] for i in range(pdf.shape[0])]
    pdf['hostgal_sqradius'] = [pdf['diaObject'].iloc[i]['hostgal_sqradius'] for i in range(pdf.shape[0])]
    pdf['hostgal_zspec'] = [pdf['diaObject'].iloc[i]['hostgal_zspec'] for i in range(pdf.shape[0])]
    pdf['hostgal_zspec_err'] = [pdf['diaObject'].iloc[i]['hostgal_zspec_err'] for i in range(pdf.shape[0])]
    pdf['mwebv'] = [pdf['diaObject'].iloc[i]['mwebv'] for i in range(pdf.shape[0])]
    pdf['mwebv_err'] = [pdf['diaObject'].iloc[i]['mwebv_err'] for i in range(pdf.shape[0])]
    pdf['z_final'] = [pdf['diaObject'].iloc[i]['z_final'] for i in range(pdf.shape[0])]
    pdf['z_final_err'] = [pdf['diaObject'].iloc[i]['z_final_err'] for i in range(pdf.shape[0])]

    for band in bands:
        pdf['hostgal_mag_' + band] = [pdf['diaObject'].iloc[i]['hostgal_mag_' + band] for i in range(pdf.shape[0])]
        pdf['hostgal_magerr_' + band] = [pdf['diaObject'].iloc[i]['hostgal_magerr_' + band] for i in range(pdf.shape[0])]
    
    for step in range(0, 101, 10):
        pdf['hostgal_zphot_q' + str(step).zfill(3)] = [pdf['diaObject'].iloc[i]['hostgal_zphot_q' + str(step).zfill(3)] for i in range(pdf.shape[0])]
    pdf['delta_t'] = [pdf.iloc[i]['MJD_DETECT_LAST'] - pdf.iloc[i]['MJD_DETECT_FIRST']
                      for i in range(pdf.shape[0])]
    pdf['rising'] = [pdf.iloc[i]['MJD_DETECT_LAST'] - pdf.iloc[i]['PEAKMJD'] <= 0 for i in range(pdf.shape[0])]


    sub = pdf[cols]

    sub.to_parquet(dir_output + 'classId=' + str(class_number) + '/' + fname[81:])