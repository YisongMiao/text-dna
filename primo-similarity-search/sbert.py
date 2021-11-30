from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import pandas as pd
import h5py
import json
import pickle
from pathlib import Path


path_sbert = '../../text-dna/data/open_sbert/'
path_query = '../../text-dna/data/queries/'


def emb_queries(fp):
    queries = {
        'plane': 'A plane is taking off.',
        'woman': 'A woman is peeling a potato.',
        'cat': 'The cat is licking a bottle.',
        'steve': 'Steve Jobs is the CEO of Apple Inc. She hold many dollars of money.',
        'CS': 'Computer science is one of the most revolutionary fields in scientific research.',
        'model': 'The all-* models where trained on all available training data (more than 1 billion training pairs) and are designed as general purpose models.',
        'church': 'The church has cracks in the top.',
        'state': 'The statue is offensive and people are mad that it is on display.',
        'symphony': 'A group of people are playing in a symphony.'
    }

    sentences_q = list(queries.values())
    sentences_index = list(queries.keys())

    embeddings = model.encode(sentences_q)

    emb_df = pd.DataFrame(
        data=embeddings,
        index=sentences_index
    )

    emb_df.to_hdf('{}'.format(fp), key='df')
    print('Dumped to {}'.format(fp))


def emb_sentences(sentences_all, index_list, dir, fp):
    Path(dir).mkdir(parents=True, exist_ok=True)
    save_fp = dir + '/' + fp

    print(dir, save_fp)

    batch_size = 256

    batch_count = int(len(sentences_all) / batch_size) + 1

    # batch_count = 2

    print('Batch count: {}'.format(batch_count))

    embedding_list = []

    for batch_num in range(batch_count):
        print('in batch {}'.format(batch_num))
        start = batch_num * batch_size
        end = min((batch_num + 1) * batch_size, len(sentences_all))
        print(start, end)
        sentence_batch = sentences_all[start: end]

        # Sentences are encoded by calling model.encode()
        embeddings = model.encode(sentence_batch)
        # numpy array, 128 * 384
        embedding_list.append(embeddings)

    embeddings_all = np.concatenate(embedding_list)
    emb_df = pd.DataFrame(
        data=embeddings_all,
        index=index_list
    )

    emb_df.to_hdf('{}'.format(save_fp), key='df')
    print('Dumped to {}'.format(save_fp))

    emb_df_read = pd.read_hdf('{}'.format(save_fp), 'df')

    print('Loading ...')
    print('Done')



if __name__ == '__main__':

    encoder_name = 'MiniLM'
    dataset_name = 'stsb'


    if encoder_name == 'MiniLM':
        model = SentenceTransformer('all-MiniLM-L6-v2').cuda()
    elif encoder_name == 'mpnet':
        model = SentenceTransformer('all-mpnet-base-v2').cuda()

    dataset_name_full = dataset_name
    if dataset_name == 'stsb':
        dataset_name_full = 'stsb_multi_mt'

    dataset_train = load_dataset(dataset_name_full, name="en", split="train")
    try:
        dataset_valid = load_dataset(dataset_name_full, name="en", split="dev")
    except:
        dataset_valid = load_dataset(dataset_name_full, name="en", split="validation")
    dataset_test = load_dataset(dataset_name_full, name="en", split="test")

    dataset = load_dataset(dataset_name_full, name="en", split="test")

    # For STSB
    if dataset_name == 'stsb':
        # ----- embed the query -----
        fp = '{}features-{}-{}.h5'.format(path_query, dataset_name, encoder_name)
        print(fp)
        emb_queries(fp)

        # ----- embed the feature vectors -----
        sentences_all_train = dataset_train['sentence1'] + dataset_train['sentence2']
        index_list_train = ['sentence1-{}'.format(i) for i in range(len(dataset_train['sentence1']))] + ['sentence2-{}'.format(i) for i in range(len(dataset_train['sentence2']))]
        dir = path_sbert + 'train-' + dataset_name + '-' + encoder_name + '/features/'
        fp = 'train.h5'
        print(dir, fp)
        emb_sentences(sentences_all_train, index_list_train, dir, fp)


        sentences_all_valid = dataset_valid['sentence1'] + dataset_valid['sentence2']
        index_list_valid = ['sentence1-{}'.format(i) for i in range(len(dataset_valid['sentence1']))] + ['sentence2-{}'.format(i) for i in range(len(dataset_valid['sentence2']))]
        dir = path_sbert + 'validation-' + dataset_name + '-' + encoder_name + '/features/'
        fp = 'validation.h5'
        print(dir, fp)
        emb_sentences(sentences_all_valid, index_list_valid, dir, fp)


        sentences_all_test = dataset_test['sentence1'] + dataset_test['sentence2']
        index_list_test = ['sentence1-{}'.format(i) for i in range(len(dataset_test['sentence1']))] + ['sentence2-{}'.format(i) for i in range(len(dataset_test['sentence2']))]
        dir = path_sbert + 'targets' + '/features/'
        fp = 'targets-{}-{}.h5'.format(dataset_name, encoder_name)
        print(dir, fp)
        emb_sentences(sentences_all_test, index_list_test, dir, fp)


    # sentences_all = dataset['premise'] + dataset['hypothesis']
    # index_list = ['premise-{}'.format(i) for i in range(len(dataset['premise']))] + ['hypothesis-{}'.format(i) for i in range(len(dataset['hypothesis']))]
    # print('The length of index_list is: {}'.format(len(index_list)))

    # d = dict()
    # for i in range(len(index_list)):
    #     d[index_list[i]] = sentences_all[i]
    #
    # with open('../dna-data/stsb-target.json', 'w') as f:
    #     json.dump(d, f, indent=4)
    #
    # with open('../dna-data/stsb-target.pickle', 'wb') as f:
    #     pickle.dump(d, f, protocol=2)
    #
    # print('Json dumped')
