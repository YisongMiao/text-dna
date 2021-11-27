PRIMO Similarity Search -- Workflow
=======================

Yisong and Yihao would like to thank the PRIMO Team for this great codebase! https://github.com/uwmisl/primo-similarity-search





## 1. SBERT

**Purpose:** We first need similar "feature embeddings" as used in the original image experiments. 

**Output Storage:** data/open_sbert

**Code:** [sbert.py](sbert.py)

There are four types of feature files:

- Train: `data/open_sbert/train-mpnet/`
- Validation: `data/open_sbert/validation-mpnet/`
- Target: `data/open_sbert/targets/features/targets-mpnet.h5`
- Queries: `data/queries/feature_seqs-mpnet.h5`

where `mpnet` is a suffix, specifying the text encoder we are currently using. 



**Data size:**

- train ~10k
- validation ~2k
- targets ~2k
- queries = 6 (in Image, they only have 3 queries. )



Note that `sbert.py` is using Python 3, NOT python 2 in the primary working package. 

We recommend running it in another environment and copy the files to the directory above 





## 2. train_encoder & train_predictor

**Purpose:** as the name per se, train the encoder & predictor!

**Output Storage:** data/model

**Code:** [train_encoder.py](train_encoder.py)

We also have  [train_predictor.py](train_predictor.py), it is used train the predictor. It does not require text data. It only requires synthesized data. But **we still need to re-train if** we want to change the dimensions (e.g. 80 -> 40).



The output is:

```python
    print 'Saving encoder'
    encoder.save('../data/models/encoder_model-mpnet.h5')

    print 'Saving predictor'
    yield_predictor.save('../data/models/predictor_model-mpnet.h5')
```





## 3. Encode_query_target

**Purpose:** as the name per se, encode the query and target from sentence embeddings 🔤 into DNA sequences 🧬

**Output Storage:** data/queries and data/targets

**Code:** [encode_query_target.py](encode_query_target.py)



Output: 

```python
# Query ...
query_features = pd.read_hdf('../data/queries/features-mpnet.h5')
query_seqs = encoder.encode_feature_seqs(query_features)
pd.DataFrame(
    query_seqs, index=query_features.index, columns=['FeatureSequence']
).to_hdf(
    '../data/queries/feature_seqs-mpnet.h5', key='df', mode='w'
)

...

# Target ...

# Memory-mapped file that caches the distances between targets and queries
dist_store = pd.HDFStore('../data/targets/query_target_dists-mpnet.h5', complevel=9, mode='w')

# Memory-mapped file that stores the DNA sequence encodings of the target features.
seq_store = pd.HDFStore('../data/targets/feature_seqs-mpnet.h5', complevel=9, mode='w')
```



## 4. Simulation

**Purpose:** as the name per se, simulate the DNA sequence of targets and queries, to see their yield (a scalar in 0~1).

**Output Storage:** data/simulation

**Code:** [simulation.py](simulation.py).



Output:

```python
result_store = pd.HDFStore('../data/simulation/extended_targets/steve-self.h5', complevel=9, mode='w')
```



Note that we need to specify what is the query here, like `steve`

```python
pairs = (target_seqs
 .rename(columns={'FeatureSequence':'target_features'})
 .assign(query_features = query_seqs.loc['steve'].FeatureSequence)
)
```



To see the queries, it is generated in [sbert.py](sbert.py).



## 5. Plot

**Purpose:** as the name per se, plot the results. Nothing particular. 

**Code:** [plot.py](plot.py).





## More about Data!

We basically have these directories under `/data`

```
305M    extended_targets
55M     metadata
2.8G    models
8.2G    open_images
127M    open_sbert
5.0M    queries
176M    sequencing
197M    simulation
98M     targets
```



For training of image (VGG based) and text (BERT based), we have the data under `open_images` and `open_sbert`  respectively. 

```
ysmiao@next-dgx1-02:~/text-dna/data$ ls open_images/
train  validation
ysmiao@next-dgx1-02:~/text-dna/data$ ls open_sbert/
train  validation
```



For simulation, we use `queries` and `targets`. Their data structure look like this:

```
ysmiao@next-dgx1-02:~/text-dna/data/queries$ du -sh *
13K     feature_seqs.h5
57K     features.h5
790K    images
ysmiao@next-dgx1-02:~/text-dna/data/targets$ du -sh *
58M     feature_seqs.h5
41M     query_target_dists.h5
```



More:

```
open_images:
total 10
drwxrwxr-x  4 ysmiao 1015  4 Nov 22 12:17 ./
drwxrwxr-x 11 ysmiao 1015 11 Nov 23 00:00 ../
drwxrwxr-x  4 ysmiao 1015  4 Nov 22 12:17 train/
drwxrwxr-x  4 ysmiao 1015  4 Nov 22 12:17 validation/

open_sbert:
total 20
drwxrwxr-x  8 ysmiao 1015  8 Nov 26 19:41 ./
drwxrwxr-x 11 ysmiao 1015 11 Nov 23 00:00 ../
drwxrwxr-x  2 ysmiao 1015  2 Nov 23 22:00 queries/
drwxrwxr-x  3 ysmiao 1015  3 Nov 23 22:07 targets/
drwxrwxr-x  4 ysmiao 1015  4 Nov 23 00:01 train/
drwxrwxr-x  3 ysmiao 1015  4 Nov 26 19:45 train-mpnet/
drwxrwxr-x  4 ysmiao 1015  4 Nov 23 00:01 validation/
drwxrwxr-x  3 ysmiao 1015  3 Nov 26 19:49 validation-mpnet/

queries:
total 4289
drwxrwxr-x  3 ysmiao 1015      10 Nov 26 20:33 ./
drwxrwxr-x 11 ysmiao 1015      11 Nov 23 00:00 ../
-rw-rw-r--  1 ysmiao 1015 2112568 Nov 26 21:40 feature_seqs.h5
-rw-rw-r--  1 ysmiao 1015 2112568 Nov 26 21:20 feature_seqs-mpnet.h5
-rw-rw-r--  1 ysmiao 1015 2112568 Nov 26 18:42 feature_seqs-text.h5
-rw-rw-r--  1 ysmiao 1015  119938 Nov 22 12:19 features.h5
-rw-rw-r--  1 ysmiao 1015   35942 Nov 26 20:28 features-mpnet.h5
-rw-rw-r--  1 ysmiao 1015   15959 Nov 23 23:43 features-text.h5
drwxrwxr-x  2 ysmiao 1015       5 Nov 22 12:19 images/
-rw-rw-r--  1 ysmiao 1015 4286236 Nov 23 23:40 targets-text.h5

sequencing:
total 21
drwxrwxr-x  9 ysmiao 1015  9 Nov 22 12:17 ./
drwxrwxr-x 11 ysmiao 1015 11 Nov 23 00:00 ../
drwxrwxr-x  2 ysmiao 1015  5 Nov 22 12:17 Run_101/
drwxrwxr-x  2 ysmiao 1015  5 Nov 22 12:17 Run_103/
drwxrwxr-x  2 ysmiao 1015  5 Nov 22 12:17 Run_104/
drwxrwxr-x  2 ysmiao 1015  5 Nov 22 12:17 Run_105/
drwxrwxr-x  2 ysmiao 1015  5 Nov 22 12:17 Run_107/
drwxrwxr-x  2 ysmiao 1015  5 Nov 22 12:17 Run_92/
drwxrwxr-x  2 ysmiao 1015  5 Nov 22 12:17 Run_98/

simulation:
total 18
drwxrwxr-x  4 ysmiao 1015  4 Nov 22 12:19 ./
drwxrwxr-x 11 ysmiao 1015 11 Nov 23 00:00 ../
drwxrwxr-x  2 ysmiao 1015  8 Nov 26 21:21 extended_targets/
drwxrwxr-x  2 ysmiao 1015  3 Nov 22 12:19 targets/

targets:
total 100176
drwxrwxr-x  2 ysmiao 1015        8 Nov 26 20:34 ./
drwxrwxr-x 11 ysmiao 1015       11 Nov 23 00:00 ../
-rw-rw-r--  1 ysmiao 1015 60286022 Nov 22 12:19 feature_seqs.h5
-rw-rw-r--  1 ysmiao 1015    49930 Nov 26 21:20 feature_seqs-mpnet.h5
-rw-rw-r--  1 ysmiao 1015    69521 Nov 26 18:42 feature_seqs-text.h5
-rw-rw-r--  1 ysmiao 1015 42021011 Nov 22 12:19 query_target_dists.h5
-rw-rw-r--  1 ysmiao 1015    67373 Nov 26 21:20 query_target_dists-mpnet.h5
-rw-rw-r--  1 ysmiao 1015    43910 Nov 26 18:42 query_target_dists-text.h5
```









## Reflection

Nov 23: Perhaps we should train the encoder model on a larger dataset (e.g. SNLI), and evaluate it on SST?







---

Following are the original readme by UW. Yihao and Yisong thank their efforts!



Setup
-----
This repository comes with a Dockerfile which allows you to reproduce our
development environment. To use it, you must have a GPU-equipped server or
workstation, and the ability to download and install
[docker](https://www.docker.com/) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Note that this environment does not include the image dataset 
([OpenImages V4](https://storage.googleapis.com/openimages/web/download_v4.html)) used for
training and experiments. This dataset is publicly available, but requires over 1 terabyte of storage space, and a
significant amount of time to download. Scripts to manage the download are included in this
repository (see [Downloading Datasets](#downloading-datasets) below).

For convenience, we have pre-processed the images with VGG16-FC2 to extract feature vectors.
These feature vectors take up about 60 gigabytes and are available for download from the 
[primo-openimages](https://github.com/uwmisl/primo-openimages) repository. 

Once you have installed docker and nvidia-docker, run the following command in
this directory to build the docker image:

```
docker build -t primo .
```

Then run the following command to start the container, which will launch a
jupyter notebook server on port 8888 (use `-p PORT` to specify a different one):

```
sudo bash docker.sh -d /path/to/primo-openimages
```

Replace `/path/to/primo-openimages` with the path to the `primo-openimages` repository.


Downloading Datasets
--------------------
The [primo-openimages](https://github.com/uwmisl/primo-openimages) repository contains
the VGG16-FC2 feature vectors for the images used in our experiments. These are sufficient
to train the encoder and perform wetlab experiments, but if you wish to view the images themselves
you will need to download the original files.

If you just want to download or view a single image,
you can use its unique identifier to look up its URL, using
[this index](https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv).

If you want to download all of the images and organize them into the same sets that we used
for our experiments, open and run [this notebook](notebooks/01_datasets/01_download.ipynb).

The code used to extract the feature vectors can be found in
[this notebook](notebooks/01_datasets/02_extract_features.ipynb).
