import tensorflow as tf
from tensorflow.keras import layers

import primo.models
import primo.datasets

import cupyck

import numpy as np

cupyck_sess = cupyck.GPUSession(max_seqlen=200, nblocks=1000, nthreads=128)

simulator = primo.models.Simulator(cupyck_sess)

print 'Loading training ... '
train_dataset = primo.datasets.OpenImagesTrain(
    '../data/open_images/train/', switch_every=10**5
)

print 'Loading validation ... '
val_dataset = primo.datasets.OpenImagesVal('../data/open_images/validation/')

def keras_batch_generator(dataset_batch_generator, sim_thresh):
    while True:
        indices, pairs = next(dataset_batch_generator)
        distances = np.sqrt(np.square(pairs[:,0,:] - pairs[:,1,:]).sum(1))
        similar = (distances < sim_thresh).astype(int)
        yield pairs, similar

# To see how this value was derived, please consult the Materials and Methods subsection under Feature Extraction section.
sim_thresh = 75
# Intuitively determined:
encoder_train_batch_size = 100
encoder_val_batch_size = 2500
predictor_train_batch_size = 1000

print 'Creating train batches'
encoder_train_batches = keras_batch_generator(
    train_dataset.balanced_pairs(encoder_train_batch_size, sim_thresh),
    sim_thresh
)

print 'Creating valid batches'
encoder_val_batches = keras_batch_generator(
    val_dataset.random_pairs(encoder_val_batch_size),
    sim_thresh
)

print 'Models ...'
predictor_train_batches = train_dataset.random_pairs(1000)

encoder = primo.models.Encoder()
yield_predictor = primo.models.Predictor('../data/models/yield-model.h5')
encoder_trainer = primo.models.EncoderTrainer(encoder, yield_predictor)

print 'Compiling models ...'
encoder_trainer.model.compile(tf.keras.optimizers.Adagrad(1e-3), 'binary_crossentropy')

print 'Start training ...'
history = encoder_trainer.model.fit_generator(
    encoder_train_batches,
    steps_per_epoch = 1000,
    epochs = 100,
    callbacks = [
        encoder_trainer.refit_predictor(
            predictor_train_batches, simulator, refit_every=1, refit_epochs=10
        )
    ],
    validation_data = encoder_val_batches,
    validation_steps = 1,
    verbose = 2
)


if __name__ == '__main__':
    print 'Done'

    print 'Saving encoder'
    encoder.save('../data/models/encoder_model.h5')

    print 'Saving predictor'
    predictor.save('../data/models/predictor_model.h5')


