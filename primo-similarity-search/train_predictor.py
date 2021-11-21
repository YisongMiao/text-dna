import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cupyck
import primo.models
import primo.tools.sequences as seqtools

cupyck_sess = cupyck.GPUSession(max_seqlen=200, nblocks=1024, nthreads=128)
simulator = primo.models.Simulator(cupyck_sess)

# Generate two random DNA sequences. First by creating a random sequence, then copying that sequence to another sequence, then mutating that second sequence at some given mutation rate, such that
# the hamming distance between the sequences (i.e. how different they are) is sampled from a uniform distribution. This means that we could expect sequence that are nearly identical, and sequence pairs that are entirely opposite to occur at approximately the same rate.
# Random_pairs: (arbitrary_sequence_1, arbitrary_sequence_2)

# Generate 5,000 of these arbitrary sequence pairs
# ...of length 80 nucleotides each (80 was chosen because that's the length of the DNA sequence's feature region in callie original paper).
random_pairs, mut_rates = seqtools.random_mutant_pairs(5000, 80)


# From here, we take the first arbitrary sequence, and its mutation (i.e. the random pairs), then we pretend that the first arbitrary sequence (i.e. at index 0) is a target DNA sequence, then we pretend the second arbitrary sequence (i.e. at index 1) is a query DNA sequences.
seq_pairs = pd.DataFrame({
    "target_features": random_pairs[:, 0],
    "query_features": random_pairs[:, 1]
})

# Use NUPACK/CUPYCK to simulate the hybdrization yields of the two sequences in the pair (this is the Pink flow in the above diagram).
sim_results = simulator.simulate(seq_pairs)

predictor = primo.models.Predictor()

onehot_seq_pairs = predictor.seq_pairs_to_onehots(seq_pairs)

history = predictor.train(onehot_seq_pairs, sim_results.duplex_yield, epochs=50, validation_split = 0.2)

pred_yield = predictor.model.predict(onehot_seq_pairs)


if __name__ == '__main__':
    print 'done'
    # OPK by Nov 21 6:19PM

