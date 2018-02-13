
"""
    in this file, we declare all of the imports and constants at once
"""

from __future__ import print_function
from __future__ import division
import tensorflow as tf
import os # kill the tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gzip
import random
import numpy as np
from subprocess import call # this will call linux commands from python code
from six.moves import urllib,xrange # urllib is used to download the dataset
from matplotlib import pyplot as plot

# data download paths
URL = 'http://yann.lecun.com/exdb/mnist/'
train_images_file = 'train-images-idx3-ubyte.gz'
train_labels_file = 'train-labels-idx1-ubyte.gz'
test_images_file = 't10k-images-idx3-ubyte.gz'
test_labels_file = 't10k-labels-idx1-ubyte.gz'

# some constants
IMAGE_INPUT_SIZE = 28 # this means that the width and height of the image are the same
CHANNELS = 1
# the original data has no eval set, just 6000 train images but we want 1000 eval images
NUM_TRAIN_IMAGES = 5000
NUM_EVAL_IMAGES = 1000
NUM_TEST_IMAGES = 1000
NUM_LABELS = 10 # ofcourse, labels are numbers from 0-9
TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 20
TEST_BATCH_SIZE = 20
NUM_EPOCHS = 10
