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

import pandas as pd
import numpy as np
import os
from copy import deepcopy

from actsnfink import *
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score

from actsnclass import DataBase
import pickle

bands = ['u', 'g', 'r', 'i', 'z', 'Y']

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

input_dir = '/media/ELAsTICC/Fink/first_year/early_SNIa/all_features/'

flist = os.listdir(input_dir)
flist.remove('.ipynb_checkpoints')

data_list = []
for fname in flist:
    data_temp = pd.read_csv(input_dir + fname, index_col=False)
    
    col_remove = []
    for colname in data_temp.keys():
        if 'Unnamed' in colname:
            col_remove.append(colname)
            
    if len(col_remove) > 0:
        data_temp.drop(columns=col_remove, inplace=True)

    data_list.append(data_temp)
    
data_pd = pd.concat(data_list, ignore_index=True)

#### train
train_dir = '/media/ELAsTICC/Fink/first_year/early_SNIa/AL/UncSampling/training_samples/'
fname_train_list = os.listdir(train_dir)

train_used = []
for fname in fname_train_list:
    data_temp = pd.read_csv(train_dir + fname, index_col=False)
    train_used.append(data_temp)
    
#### test
test_dir = '/media/ELAsTICC/Fink/first_year/early_SNIa/AL/UncSampling/test_samples/'
fname_test_list = os.listdir(test_dir)

test_used = []
for fname in fname_test_list:
    data_temp = pd.read_csv(test_dir + fname, index_col=False)
    test_used.append(data_temp)

data_used = pd.concat(train_used + test_used, ignore_index=True)

data_free = data_pd[~np.isin(data_pd['alertId'].values, data_used['id'].values)]
features_names_rep = ['classId'] + list(data_free.keys())

data_unique = data_free.drop_duplicates(subset=features_names_rep, keep='first')

# separate train and test per object
objects = np.unique(data_unique['diaObjectId'].values)

objects_train = np.random.choice(objects, size=int(len(objects)/2), replace=False)
train_flag = np.isin(data_unique['diaObjectId'].values, objects_train)

data_train_all = data_unique[train_flag]
data_test_all = data_unique[~train_flag]

features_names = list(data_train_all.keys())[3:]

# initiate object
data = DataBase()
data.features_names = features_names
data.metadata_names = ['diaObjectId', 'id', 'classId', 'type']

# separate ias and nonias
ia_flag = data_train_all['classId'].values == 111
train_ia = data_train_all[ia_flag]
train_nonia = data_train_all[~ia_flag]

# build training
n_train = 20
train_pd = pd.concat([train_ia.sample(n=int(n_train/2), replace=False), train_nonia.sample(n=int(n_train/2), replace=False)])

train_pd['type'] = [taxonomy[item] for item in train_pd['classId'].values]
data.train_metadata = deepcopy(train_pd[list(data_train_all.keys())[:3] + ['type']])
data.train_features = deepcopy(train_pd[list(data_train_all.keys())[3:]].values)
train_labels = data.train_metadata['classId'].values == 111
data.train_labels = train_labels.astype(int)
data.train_metadata.rename(columns={'alertId':'id'}, inplace=True)

current_test = data_test_all.sample(n=100000, replace=False)
current_test['type'] = [taxonomy[item] for item in current_test['classId'].values]
data.test_metadata = deepcopy(current_test[list(data_test_all.keys())[:3] + ['type']])
test_labels = data.test_metadata['classId'].values == 111
data.test_features= current_test[data.features_names].values
data.test_labels = test_labels.astype(int)
data.test_metadata.rename(columns={'alertId':'id'}, inplace=True)

data.queryable_ids = data.test_metadata['id'].values

screen = True
classifier = 'RandomForest'
nest = 30
seed = 42
max_depth = 30
n_jobs = 20
strategy = 'UncSampling'
batch = 1
output_metrics_file = '/media/ELAsTICC/Fink/first_year/early_SNIa/AL/UncSampling/metrics/metric_' + \
                      strategy + '.csv'
output_queried_file = '/media/ELAsTICC/Fink/first_year/early_SNIa/AL/UncSampling/queries/queried_' + \
                      strategy + '.csv'

for loop in range(1000):
    # classify
    data.classify(method=classifier, seed=seed, n_est=nest, max_depth=max_depth)
    
    # calculate metrics
    data.evaluate_classification(screen=screen)

    # choose object to query
    indx = data.make_query(strategy=strategy, batch=batch, screen=screen)

    # update training and test samples
    data.update_samples(indx, loop=loop)

    # save metrics for current state
    data.save_metrics(loop=loop, output_metrics_file=output_metrics_file,
                      batch=batch, epoch=loop)

    # save query sample to file
    data.save_queried_sample(output_queried_file, loop=loop,
                             full_sample=False)
    
    
final_train = pd.concat([data.train_metadata, pd.DataFrame(data.train_features, columns=features_names)], axis=1)
final_train.to_csv('/media/ELAsTICC/Fink/first_year/early_SNIa/AL/UncSampling/training_samples/train_1020_v' + 
                   str(len(fname_train_list) + 1) + '.csv')

final_test = pd.concat([data.test_metadata, pd.DataFrame(data.test_features, columns=features_names)], axis=1)
final_test.to_csv('/media/ELAsTICC/Fink/first_year/early_SNIa/AL/UncSampling/test_samples/test_' + \
                  str(data.test_metadata.shape[0]) +'_v' + str(len(fname_train_list) + 1) + '.csv', index=False)

nest = 30
seed = 42
max_depth = 30
n_jobs = 20


clf = RandomForestClassifier(n_estimators=nest, random_state=seed,
                             max_depth=max_depth, n_jobs=n_jobs)
clf.fit(data.train_features, data.train_labels)

pred = clf.predict(data.test_features)
print('acc: ', clf.score(data.test_features, data.test_labels))
print('pur:', sum(data.test_labels[pred == 1])/sum(pred))

filename = 'model_AL_1020_v' +  str(len(fname_train_list) + 1) + '.pkl'
pickle.dump(clf, open(filename, 'wb'))