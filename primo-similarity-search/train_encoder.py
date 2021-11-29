import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import primo.models
import primo.datasets

import cupyck

import numpy as np

cupyck_sess = cupyck.GPUSession(max_seqlen=200, nblocks=1000, nthreads=128)

simulator = primo.models.Simulator(cupyck_sess)



# # TODO: Yisong: Config for Image ...
# print 'Loading training ... '
# train_dataset = primo.datasets.OpenImagesTrain(
#     '../data/open_images/train/', switch_every=10**5
# )
#
# print 'Loading validation ... '
# val_dataset = primo.datasets.OpenImagesVal('../data/open_images/validation/')
#
# def keras_batch_generator(dataset_batch_generator, sim_thresh):
#     while True:
#         indices, pairs = next(dataset_batch_generator)
#         distances = np.sqrt(np.square(pairs[:,0,:] - pairs[:,1,:]).sum(1))
#         similar = (distances < sim_thresh).astype(int)
#         yield pairs, similar
#
# # To see how this value was derived, please consult the Materials and Methods subsection under Feature Extraction section.
# sim_thresh = 75
# # Intuitively determined:
# encoder_train_batch_size = 100
# encoder_val_batch_size = 2500
# predictor_train_batch_size = 1000


# TODO: Yisong: Config for Text ...

# print 'Loading training ... '
# train_dataset = primo.datasets.OpenImagesTrain(
#     '../data/open_sbert/train/', switch_every=10**5
# )
# print 'Loading validation ... '
# val_dataset = primo.datasets.OpenImagesVal('../data/open_sbert/validation/')

print 'Loading training ... '
train_dataset = primo.datasets.OpenImagesTrain(
    '../data/open_sbert/train-snli/', switch_every=10**5
)
print 'Loading validation ... '
val_dataset = primo.datasets.OpenImagesVal('../data/open_sbert/validation-snli/')


def keras_batch_generator(dataset_batch_generator, sim_thresh):
    while True:
        indices, pairs = next(dataset_batch_generator)
        distances = np.sqrt(np.square(pairs[:,0,:] - pairs[:,1,:]).sum(1))
        similar = (distances < sim_thresh).astype(int)
        yield pairs, similar

# To see how this value was derived, please consult the Materials and Methods subsection under Feature Extraction section.
sim_thresh = 1.33
# Intuitively determined:
encoder_train_batch_size = 100
encoder_val_batch_size = 2500
predictor_train_batch_size = 1000


print 'Defining train batches creation'
encoder_train_batches = keras_batch_generator(
    train_dataset.balanced_pairs(encoder_train_batch_size, sim_thresh),
    sim_thresh
)

print 'Creating valid batches'
encoder_val_batches = keras_batch_generator(
    val_dataset.random_pairs(encoder_val_batch_size),
    sim_thresh
)

# TODO: Yisong, alternate it into balanced pair?
encoder_val_batches = keras_batch_generator(
    val_dataset.random_pairs(encoder_val_batch_size),
    sim_thresh
)

print 'Models ...'
predictor_train_batches = train_dataset.random_pairs(1000)

encoder = primo.models.Encoder()
yield_predictor = primo.models.Predictor('../data/models/yield-model.h5')
# yield_predictor = primo.models.Predictor('../data/models/yield-predictor-original-40.h5')
encoder_trainer = primo.models.EncoderTrainer(encoder, yield_predictor)

print 'Compiling models ...'
encoder_trainer.model.compile(optimizer=tf.keras.optimizers.Adagrad(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

filepath = "../data/models/encoder_model-snli-{epoch:02d}-{val_loss:.2f}.hdf5"
mc = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='min')


# TODO Yisong: add val_acc!
print 'Start training ...'
history = encoder_trainer.model.fit_generator(
    encoder_train_batches,
    # steps_per_epoch = 1000,
    steps_per_epoch = 100, # TODO: Yisong, change for text ...
    epochs = 100,
    callbacks = [
        encoder_trainer.refit_predictor(
            predictor_train_batches, simulator, refit_every=1, refit_epochs=10
        ),
        es,
        mc
    ],
    validation_data = encoder_val_batches,
    validation_steps = 1,
    verbose = 2
)


if __name__ == '__main__':
    print 'Done'

    print 'Saving encoder'
    encoder.save('../data/models/encoder_model-snli.h5')

    print 'Saving predictor'
    yield_predictor.save('../data/models/predictor_model-snli.h5')

