
"""
    this is our class to download and extract the dataset from yann lecun's website
    and return the dataset as np array, we shall also make all of our imports in this single file
"""

from __future__ import print_function
from __future__ import division
from imports_module import*

class Download_and_Extract(object):
    "this class downloads the dataset from yann lecun's site"
    def __init__(self,main_url=None,data_folder_name=os.getcwd()):
        "just creates a dataset folder if needed"
        self.main_url = main_url
        self.folder = os.path.join(os.getcwd(),data_folder_name)
        if not tf.gfile.Exists(self.folder):
            call('mkdir {}'.format(self.folder),shell=True)
        return

    def download_if_needed(self,_file=None):
        "check if it does not exit and download it if needed"
        full_path = os.path.join(self.folder,_file)
        if not tf.gfile.Exists(filename=full_path):
            print('log: downloading {} into {} now!'.format(self.main_url+_file,self.folder))
            full_path = os.path.join(self.folder,_file)
            # test = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
            filepath, _ = urllib.request.urlretrieve(self.main_url+_file,full_path)
            with tf.gfile.GFile(full_path, mode='r') as downloaded:
                print('log: We have successfully downloaded {} MBs of {}'.format(downloaded.size()/1024/1024,_file))
        else:
            print('log: filename = {} found ({}).'.format(_file,full_path))

    def py_extract(self,what='Nothing',_file=None,class_='train',NUM_IMAGES=1):
        save_arr_name = os.path.join(self.folder,'{}_{}.npy'.format(class_,what))
        if not tf.gfile.Exists(filename=save_arr_name):
            with gzip.open(self.folder+'/'+_file) as fileObj:
                if what is 'images':
                    fileObj.read(size=16)
                    # print(self.folder+'/'+_file)
                    bufferObj = fileObj.read(size=IMAGE_INPUT_SIZE*IMAGE_INPUT_SIZE*CHANNELS*NUM_IMAGES)
                    npdata = np.frombuffer(bufferObj, dtype=np.int8).astype(np.float64)
                    # print(npdata.shape)
                    npdata = npdata.reshape(NUM_IMAGES,IMAGE_INPUT_SIZE,IMAGE_INPUT_SIZE,CHANNELS)
                    np.save(save_arr_name, npdata, allow_pickle=True, fix_imports=True)
                    print('{} extracted and saved as {}'.format(_file,save_arr_name))
                    return npdata
                elif what is 'labels':
                    fileObj.read(size=8)
                    bufferObj = fileObj.read(size=NUM_IMAGES)
                    npdata = np.frombuffer(bufferObj, dtype=np.int8).astype(np.int8)
                    # convert this into one-hot array
                    one_hot_arr = np.zeros(shape=(npdata.shape[0],NUM_LABELS))
                    one_hot_arr[range(npdata.shape[0]), npdata] = 1
                    np.save(save_arr_name, one_hot_arr, allow_pickle=True, fix_imports=True)
                    print('{} extracted and saved as {}'.format(_file,save_arr_name))
                    return one_hot_arr
                else:
                    print('log: unknown argument: {} passed'.format(what))
                    return
        else:
            print('log: {} already exists. loading now.'.format(save_arr_name))
            return np.load(save_arr_name)
