# https://github.com/YisongMiao/text-dna/blob/main/primo-similarity-search/notebooks/01_datasets/02_extract_features.ipynb

import numpy as np
import tensorflow as tf
import pandas as pd

import primo.models
import primo.datasets

from tqdm.notebook import tqdm


encoder = primo.models.Encoder('../data/models/encoder_model-self-text-best.h5')

query_features = pd.read_hdf('../data/queries/features-text.h5')

query_seqs = encoder.encode_feature_seqs(query_features)
pd.DataFrame(
    query_seqs, index=query_features.index, columns=['FeatureSequence']
).to_hdf(
    '../data/queries/feature_seqs-text.h5', key='df', mode='w'
)

# Memory-mapped file that caches the distances between targets and queries
dist_store = pd.HDFStore('../data/targets/query_target_dists-text.h5', complevel=9, mode='w')

# Memory-mapped file that stores the DNA sequence encodings of the target features.
seq_store = pd.HDFStore('../data/targets/feature_seqs-text.h5', complevel=9, mode='w')

try:
    # Target images are split up across 16 files.
    # Because these files are so large, and can't all be stored into memory on a single machine, there's some low-level memory-management wizardly happening below.
    # prefixes = ["%x" % i for i in range(16)]
    # for prefix in tqdm(prefixes):
    #     target_features = pd.read_hdf('/tf/open_images/targets/features/targets_%s.h5' % prefix)
    #
    #     # Dictionary that maps queries to euclidean distances for 1every pairing of query and target.
    #     distances = {}
    #     for query_id, query in query_features.iterrows():
    #         # Calculuate the Euclidean distance between each query and target.
    #         distances[query_id] = np.sqrt(np.square(target_features.values - query.values).sum(1))
    #
    #     df = pd.DataFrame(distances, index=target_features.index)
    #     dist_store.append('df', df)
    #
    #     # Low-level memory mangement
    #     del df, distances
    #
    #     target_seqs = encoder.encode_feature_seqs(target_features)
    #     df = pd.DataFrame(target_seqs, index=target_features.index, columns=['FeatureSequence'])
    #     seq_store.append('df', df)
    #     del df, target_seqs
    #
    #     del target_features

    target_features = pd.read_hdf('../data/open_sbert/targets/features/targets-text.h5')

    # Dictionary that maps queries to euclidean distances for 1every pairing of query and target.
    distances = {}
    for query_id, query in query_features.iterrows():
        # Calculuate the Euclidean distance between each query and target.
        distances[query_id] = np.sqrt(np.square(target_features.values - query.values).sum(1))

    df = pd.DataFrame(distances, index=[str(i) for i in list(target_features.index)])
    dist_store.append('df', df)
    target_seqs = encoder.encode_feature_seqs(target_features)
    df = pd.DataFrame(target_seqs, index=[str(i) for i in list(target_features.index)], columns=['FeatureSequence'])
    seq_store.append('df', df)
    print 'Cool'

    # Low-level memory mangement
    # deleted since our text data is small.


finally:
    dist_store.close()
    seq_store.close()


if __name__ == '__main__':
    print 'Done'
