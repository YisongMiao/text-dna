import numpy as np
import tensorflow as tf
import pandas as pd

import primo.models
import primo.datasets

from tqdm.notebook import tqdm
import cupyck
import time

encoder = primo.models.Encoder('../data/models/encoder_model-self-text.h5')

query_features = pd.read_hdf('../data/queries/features-text.h5')

query_seqs = encoder.encode_feature_seqs(query_features)

pd.DataFrame(
    query_seqs, index=query_features.index, columns=['FeatureSequence']
).to_hdf(
    '../data/queries/feature_seqs.h5', key='df', mode='w'
)

# hosts = [
#     ("localhost", 2047),
# ]
# client = cupyck.Client(hosts)

cupyck_sess = cupyck.GPUSession(max_seqlen=200, nblocks=1024, nthreads=128)

simulator = primo.models.Simulator(cupyck_sess)


target_seqs = pd.read_hdf('../data/extended_targets/feature_seqs.h5')
# target_seqs = pd.read_hdf('../data/targets/feature_seqs-text.h5')

# (5577710, 1)
# Column Name: 'FeatureSequence'

query_seqs = pd.read_hdf('../data/queries/feature_seqs.h5')
# query_seqs = pd.read_hdf('../data/queries/feature_seqs-text.h5')
# Shape: (3, 1)
# Index([u'callie_janelle', u'luis_lego', u'yuan_taipei'], dtype='object')
# OK. I should also design such structure.

# callie_janelle -> plain

pairs = (target_seqs
 .rename(columns={'FeatureSequence':'target_features'})
 .assign(query_features = query_seqs.loc['plain'].FeatureSequence)
)

# 4,000 here is just a memory-management batch size so that each progress chunk reports period of time.
split_size = 4000
# split_size = 40
nsplits = len(pairs) / split_size
# 5577710 / 4000 = 1394

# splits = np.array_split(pairs, nsplits)
#
# # TODO For debugging usage.
#
# splits = splits[:2]

splits = pairs[:4000]

result_store = pd.HDFStore('../data/simulation/extended_targets/replicate-self.h5', complevel=9, mode='w')

try:
    # results = simulator.simulate(splits)
    # result_store.append('df', results[['duplex_yield']])

    print 'Starting simulating ...'
    start = time.time()
    results = simulator.simulate(splits)
    print 'Results for one split'
    end = time.time()
    print 'Elapsed time is: {}'.format(end - start)
    print 'Result shape: {}'.format(results.shape)
    result_store.append('df', results[['duplex_yield']])

    # Elapsed
    # time is: 130.662931204
    # Result
    # shape: (4002, 5)

    # for split in tqdm(splits):
    #     results = simulator.simulate(split)
    #     print 'Results for one split'
    #     result_store.append('df', results[['duplex_yield']])
finally:
    result_store.close()


if __name__ == '__main__':
    print 'Done'
