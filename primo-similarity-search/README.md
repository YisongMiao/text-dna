PRIMO Similarity Search
=======================

Yisong and Yihao would like to thank the PRIMO Team for this great codebase! https://github.com/uwmisl/primo-similarity-search



## Dataset Flow

OK, we understand that the data structure. 



We basically have these directories under `/data`

```
305M    extended_targets
55M     metadata
105M    models
8.2G    open_images
39M     open_sbert
860K    queries
176M    sequencing
82M     simulation
98M     targets
```



For training of image (VGG based) and text (BERT based), we have the data under `open_images` and `open_sbert`  respectively. 

```
ysmiao@next-dgx1-02:~/text-dna/data$ ls open_images/
train  validation
ysmiao@next-dgx1-02:~/text-dna/data$ ls open_sbert/
train  validation
```



For the preparation of `open_sbert`, the text is embedded using [sbert.py](sbert.py)



The training of predictor is almost the same as image dataset. Because it only takes synthesized data. See [train_predictor.py](train_predictor.py)



The training of encoder is where changes happen. We have to customize many things. See [train_encoder.py](train_encoder.py)





Before simulation we need to prepare the DNA sequences using our trained encoder. 

The code is [encode_query_target.py](encode_query_target.py)







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



The code is [simulation.py](simulation.py).





## Reflection

Nov 23: Perhaps we should train the encoder model on a larger dataset (e.g. SNLI), and evaluate it on SST?



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
