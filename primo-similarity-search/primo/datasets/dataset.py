import abc
import numpy as np

class Dataset(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def random_pairs(self, batch_size):
        pass

    def balanced_pairs(self, batch_size, sim_thresh):

        pair_generator = self.random_pairs(batch_size)

        while True:

            n_batch = 0
            batch_ids  = []
            batch_vals = []

            while n_batch < batch_size:
                chunk_ids, chunk_vals = next(pair_generator)

                # TODO: Yisong: OK. It can be challenging. Perhaps we need to change.
                # TODO: They use a completely random strategy. Ours is different. Text dataset comes with pairs already. It means that we need to shuffle them ...
                # TODO: Yisong: we might need to change the sampling strategy. Their positive ratio is 8 %.

                distances = np.sqrt(
                    np.square(chunk_vals[:,0] - chunk_vals[:,1]).sum(1)
                )

                # Now text: [1.4136052 1.4247218 1.2709999 1.4636108 1.3038169 1.419093  1.3694252, 1.4009144 1.3632911 1.4092997 1.4480445 1.3217496 1.4483482 1.4237213, 1.4105431 1.4405496 1.3194741 1.3711942 1.2937737 1.462481  1.392356, 1.3243446 1.4639838 1.4238396 1.3435665 1.3

                # TODO Better distancing function is needed.
                # TODO Try cosine, try manhattan, etc ...


                similar = distances <= sim_thresh
                n_sim = similar.sum()

                batch_ids.extend([
                    chunk_ids[similar],
                    chunk_ids[~similar][:n_sim]
                ])

                batch_vals.extend([
                    chunk_vals[similar],
                    chunk_vals[~similar][:n_sim]
                ])

                n_batch += 2 * n_sim

            batch_ids = np.concatenate(batch_ids)
            batch_vals = np.concatenate(batch_vals)

            perm = np.random.permutation(len(batch_vals))[:batch_size]

            yield batch_ids[perm], batch_vals[perm]


class Static(Dataset):

    def __init__(self, X):
        self.X = X

    def random_pairs(self, batch_size):
        n,d = self.X.shape
        while True:
            pairs = np.random.permutation(n)[:batch_size*2].reshape(-1,2)
            yield pairs, self.X[pairs]



