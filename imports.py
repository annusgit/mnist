
" in this file, we declare all of the imports at once"

def __init__():
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
    # from matplotlib import pyplot as plot
    print('log: all imports successful!')
