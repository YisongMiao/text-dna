import os
import sys
import numpy as np
import pandas as pd

from PIL import Image

from dataset import Dataset

class OpenImagesTrain(Dataset):
    def __init__(self, path, switch_every = 1000):
        self.path = path
        self.switch_every = switch_every

    def get_images(self, img_ids):
        image_dir = os.path.join(self.path, 'images')

        for img_id in img_ids:
            img_path = os.path.join(
                image_dir, '%s/%s.jpg' % (img_id[:2], img_id)
            )
            image = Image.open(img_path)
            image.load()
            yield image

    def random_pairs(self, batch_size):

        feature_dir = os.path.join(self.path, 'features')
        files = os.listdir(feature_dir)

        while True:

            # f_a, f_b = np.random.choice(files, 2, replace=False)
            f_a, f_b = np.random.choice(files, 2, replace=True)

            # TODO Yisong
            # Seems like we should change here!

            # switching to train_0.h5 and train_0.h5. I see.
            sys.stdout.write("switching to %s and %s\n" % (f_a, f_b))

            # TODO Yisong f_a and f_b should not necessairily have same size!

            df1 = pd.read_hdf(os.path.join(feature_dir, f_a))
            # 71971, 4096
            df2 = pd.read_hdf(os.path.join(feature_dir, f_b))
            # 63646, 4096

            df = pd.concat([df1, df2])
            n = len(df)

            # self.switch_every = 100000
            for _ in range(self.switch_every):

                pairs = np.random.permutation(n)[:batch_size*2].reshape(-1,2)
                # 100, 2 [[ 55987  91161], [ 68089  27330], [ 11103 127068], [ 18037  99012], [113368  47732], [112352 115982], [ 53673  16674], [ 21444 101533], [102225  61068], [ 22909  88254], [103435  84111], [ 46448 130563], [ 97314  61850], [  6755  50637], [ 28698 111927],


                # TODO: Yisong, OK, I understand index is inserted here.
                yield df.index.values[pairs], df.values[pairs]

class OpenImagesVal(Dataset):
    
    def __init__(self, val_path):
        feature_path = os.path.join(val_path, 'features/validation.h5')
        self.df = pd.read_hdf(feature_path)
        print 'Done loading validation h5'
        
    def random_pairs(self, batch_size):
        n = len(self.df)
        while True:
            pairs = np.random.permutation(n)[:batch_size*2].reshape(-1,2)
            yield self.df.index.values[pairs], self.df.values[pairs]