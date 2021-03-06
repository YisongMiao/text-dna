# https://github.com/YisongMiao/text-dna/blob/main/primo-similarity-search/notebooks/01_datasets/02_extract_features.ipynb

import numpy as np
import tensorflow as tf
import pandas as pd

import primo.models
import primo.datasets

from tqdm.notebook import tqdm
from tensorflow.keras.models import load_model



if __name__ == '__main__':

    # encoder_name = 'MiniLM'
    # dataset_name = 'stsb'

    encoder_name = 'mpnet'
    dataset_name = 'stsb'

    dataset_name_full = dataset_name
    if dataset_name == 'stsb':
        dataset_name_full = 'stsb_multi_mt'

    model_save_dir = '../data/models/{}-{}'.format(dataset_name, encoder_name)

    for epoch in range(0, 16):

        fp = model_save_dir + '/' + '{}.h5'.format(epoch)

        print '----- \nEncoder loaded from {}\n------\n'.format(fp)

        encoder = primo.models.Encoder(fp)

        query_features = pd.read_hdf('../data/queries/features-{}-{}.h5'.format(dataset_name, encoder_name))
        query_seqs = encoder.encode_feature_seqs(query_features)
        pd.DataFrame(
            query_seqs, index=query_features.index, columns=['FeatureSequence']
        ).to_hdf(
            '../data/queries/feature_seqs-{}-{}-{}.h5'.format(dataset_name, encoder_name, epoch), key='df', mode='w'
        )

        # Memory-mapped file that caches the distances between targets and queries
        dist_store = pd.HDFStore('../data/targets/query_target_dists-{}-{}-{}.h5'.format(dataset_name, encoder_name, epoch), complevel=9, mode='w')

        # Memory-mapped file that stores the DNA sequence encodings of the target features.
        seq_store = pd.HDFStore('../data/targets/feature_seqs-{}-{}-{}.h5'.format(dataset_name, encoder_name, epoch), complevel=9, mode='w')

        try:

            target_features = pd.read_hdf('../data/open_sbert/targets/features/targets-{}-{}.h5'.format(dataset_name, encoder_name))

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


        finally:
            dist_store.close()
            seq_store.close()
            print 'Closing file'