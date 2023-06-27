# Copyright 2023 Fink Software
# Author: Emille Ishida
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://opensource.org/licenses/mit-license.php
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This file contains routines to feature extract ELAsTiCC objects
including metadata.
"""

import numpy as np
import os
import pandas as pd

from copy import deepcopy
from actsnfink.classifier_sigmoid import get_sigmoid_features_elasticc_perfilter

from progressbar import progressbar


### Global variables

# LSST filters
bands = ['u', 'g', 'r', 'i', 'z', 'Y']

# sigmoid features
features_names = ['a_','b_','c_','snratio_','mse_','nrise_']

# join all features names
features_names_filter = []
for f in bands:
    for feature in features_names:
        features_names_filter.append(feature + f)
        
# columns names from the downloaded test data, separate for use purposes
cols = ['alertId', 'classId', 'cmidPointTai', 'cpsFlux', 'cpsFluxErr', 'cfilterName']

extra_cols = ['diaObjectId', 'hostgal_dec', 'hostgal_ra', 'hostgal_snsep', 'hostgal_zphot', 'hostgal_zphot_err', 'ra', 'decl']

columns_to_keep =  ['diaObjectId', 'alertId', 'classId', 'hostgal_dec', 'hostgal_ra', 'hostgal_snsep', 'hostgal_zphot', 'hostgal_zphot_err', 'ra', 'decl']

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

#################################################################
##  user choices

class_code = '113'

dirname = '/media/ELAsTICC/Fink/first_year/ftransfer_elasticc_2023-02-15_946675/' + \
        'classId=' + class_code + '/'

output_fname = '/media/ELAsTICC/Fink/first_year/early_SNIa/all_features/class_' + class_code +'.csv'

#################################################################

# read all files for a given type
flist = os.listdir(dirname)

# read raw data
data = []

print('Reading data ...')
for fname in progressbar(flist):

    fname = dirname + fname
    pdf = pd.read_parquet(fname)
    pdf['classId'] = int(class_code)

    pdf['cpsFlux'] = pdf[['diaSource', 'prvDiaForcedSources']]\
            .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'psFlux'), axis=1)
    pdf['cpsFluxErr'] = pdf[['diaSource', 'prvDiaForcedSources']]\
            .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'psFluxErr'), axis=1)
    pdf['cfilterName'] = pdf[['diaSource', 'prvDiaForcedSources']]\
            .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'filterName'), axis=1)
    pdf['cmidPointTai'] = pdf[['diaSource', 'prvDiaForcedSources']]\
            .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'midPointTai'), axis=1)

    sub = deepcopy(pdf[cols])

    for name in extra_cols:
        sub[name] = [pdf['diaObject'][i][name] for i in range(len(pdf['diaObject']))]
        
    if sub.shape[0] > 0:
        data.append(sub)

data_pd = pd.concat(data, ignore_index=True)

data_pd.rename(columns={'cmidPointTai': 'MJD', 'cpsFlux': 'FLUXCAL', 'cpsFluxErr':'FLUXCALERR',
                       'cfilterName':'FLT'}, 
               inplace=True)

features_list = []

print('Extracting features ...')
for i in progressbar(range(data_pd.shape[0])):
    
    lc = pd.DataFrame()
    for par in ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']:
        lc[par] = data_pd.iloc[i][par]

    features = get_sigmoid_features_elasticc_perfilter(lc, list_filters=bands)
    
    if sum(np.array([features[k * 6 + 1] for k in range(len(bands))]) == 0) < 6:
        pars_temp = [data_pd.iloc[i][[par]].values[0] for par in columns_to_keep]
        
        for number in features:
            pars_temp.append(number)
            
        features_list.append(pars_temp)
        
features_pd = pd.DataFrame(features_list, 
                           columns=columns_to_keep + features_names_filter)

if features_pd.shape[0] > 0:
    features_pd.to_csv(output_fname, index=False)