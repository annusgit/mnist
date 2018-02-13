
"""
    this is my first test convolutional neural network using tensorflow
"""

from __future__ import print_function
from __future__ import division

from data_manager import*
from conv_net import*
from imports_module import*
print('log: all imports successful!')


def main(_):
    "We'll start by downloading the required dataset of mnist handwritten digits"
    manager = Download_and_Extract(main_url=URL, data_folder_name='mnist_dataset')
    manager.download_if_needed(_file=train_images_file)
    manager.download_if_needed(_file=train_labels_file)
    manager.download_if_needed(_file=test_images_file)
    manager.download_if_needed(_file=test_labels_file)

    "now let's extract these files if needed and load them"
    train_images = manager.py_extract(what='images',_file=train_images_file,
                                      NUM_IMAGES=NUM_TRAIN_IMAGES+NUM_EVAL_IMAGES,class_='train')
    train_labels = manager.py_extract(what='labels',_file=train_labels_file,
                                      NUM_IMAGES=NUM_TRAIN_IMAGES+NUM_EVAL_IMAGES,class_='train')

    "get some eval images"
    eval_images = train_images[0:NUM_EVAL_IMAGES, ...]
    train_images = train_images[NUM_EVAL_IMAGES:, ...]
    eval_labels = train_labels[0:NUM_EVAL_IMAGES, ...]
    train_labels = train_labels[NUM_EVAL_IMAGES:, ...]

    test_images = manager.py_extract(what='images',_file=test_images_file, NUM_IMAGES=NUM_TEST_IMAGES,class_='test')
    test_labels = manager.py_extract(what='labels',_file=test_labels_file, NUM_IMAGES=NUM_TEST_IMAGES,class_='test')
    print('log: train_images = {}'.format(train_images.shape))
    print('log: train_labels = {}'.format(train_labels.shape))
    print('log: eval_images = {}'.format(eval_images.shape))
    print('log: eval_labels = {}'.format(eval_labels.shape))
    print('log: test_images = {}'.format(test_images.shape))
    print('log: test_labels = {}'.format(test_labels.shape))
    "let's visualize one image from the train data"
    this_image = np.random.randint(0,NUM_TEST_IMAGES)
    one_image = train_images[this_image,:]
    one_label = train_labels[this_image,:]
    one_image = np.array(one_image,dtype=np.int8)
    one_image = np.reshape(one_image, (28,28))
    plot.figure("Is this '{}'?".format(np.argmax(one_label, axis=0)))
    plot.imshow(one_image, cmap='gray')
    plot.show()

    # so here is our full data set
    train_data = (train_images, train_labels)
    eval_data = (eval_images, eval_labels)
    test_data = (test_images, test_labels)

    "this is our main model"
    conv_net = ConvNet(train_data=train_data, eval_data=eval_data, test_data=test_data,
                       batch_size=TRAIN_BATCH_SIZE)
    with tf.Session() as sess:
        # create a model graph
        conv_net.create_model(learn_rate=8e-3, reg_factor=0)
        print('log: model created!')
        sess.run(tf.global_variables_initializer())
        conv_net.saver.restore(sess=sess, save_path='mnist_trained_models/model_final.ckpt')
        conv_net.train(sess=sess, model_folder='mnist_trained_models', save_after=200, summary_dir='mnist_summaries')
        # save final model
        conv_net.save_model(session=sess,model_folder='mnist_trained_models',step='final')

        # get the overall scores
        conv_net.over_all_train_acc(sess=sess, model_path='mnist_trained_models/model_final.ckpt')
        # also get the eval and test accuracy
        conv_net.eval_model(sess)
        conv_net.test_model(sess)

    return None

if __name__ == '__main__':
    tf.app.run(main=main, argv=None)







