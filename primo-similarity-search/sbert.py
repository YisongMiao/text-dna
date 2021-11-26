from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import pandas as pd
import h5py


def emb_queries():
    queries = {
        'plane': 'A plane is taking off.',
        'woman': 'A woman is peeling a potato.',
        'cat': 'The cat is licking a bottle.',
        'steve': 'Steve Jobs is the CEO of Apple Inc. She hold many dollars of money.',
        'CS': 'Computer science is one of the most revolutionary fields in scientific research.',
        'model': 'The all-* models where trained on all available training data (more than 1 billion training pairs) and are designed as general purpose models.'
    }

    sentences_q = list(queries.values())
    sentences_index = list(queries.keys())

    embeddings = model.encode(sentences_q)

    emb_df = pd.DataFrame(
        data=embeddings,
        index=sentences_index
    )

    fp = 'features-mpnet.h5'
    emb_df.to_hdf('../dna-data/{}'.format(fp), key='df')
    print('Dumped to {}'.format(fp))


def emb_sentences(sentences_all, index_list):
    batch_size = 128

    batch_count = int(len(sentences_all) / batch_size) + 1

    # batch_count = 2

    print('Batch count: {}'.format(batch_count))

    embedding_list = []

    prefix = 'train'
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

        # Print the embeddings
        # for sentence, embedding in zip(sentence_batch, embeddings):
        #     print("Sentence:", sentence)
        #     print("Embedding:", embedding)
        #     print("")

        # print('Done one batch')

    embeddings_all = np.concatenate(embedding_list)
    emb_df = pd.DataFrame(
        data=embeddings_all,
        index=index_list
    )

    fp = 'targets-mpnet.h5'
    emb_df.to_hdf('../dna-data/{}'.format(fp), key='df')
    print('Dumped to {}'.format(fp))

    emb_df_read = pd.read_hdf('../dna-data/{}'.format(fp), 'df')

    print('Loading ...')
    print('Done')



if __name__ == '__main__':
    dataset = load_dataset("stsb_multi_mt", name="en", split="test")
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-mpnet-base-v2')

    # dataset.num_rows
    # dataset['sentence1'] dataset['sentence2'] dataset['similarity_score']


    # Our sentences we like to encode
    # sentences = ['This framework generates embeddings for each input sentence',
    #              'Sentences are passed as a list of string.',
    #              'The quick brown fox jumps over the lazy dog.']

    sentences_all = dataset['sentence1'] + dataset['sentence2']

    index_list = ['sentence1-{}'.format(i) for i in range(len(dataset['sentence1']))] + ['sentence2-{}'.format(i) for i in range(len(dataset['sentence2']))]

    emb_sentences(sentences_all, index_list)
    # emb_queries()