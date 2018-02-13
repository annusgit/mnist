
"""
    here we shall define our convolutional neural net's class
"""

from __future__ import print_function
from __future__ import division
from imports_module import*

class ConvNet(object):
    """DOC: this is our first nn class for convnet."""
    def __init__(self, train_data, eval_data, test_data, batch_size=TRAIN_BATCH_SIZE):
        " this function just creates all the variables as weights as biases"
        # 'for filters: [x, y, no_of_channels, no_of_filters]'
        # 'for placeholder: [batch_size, y, x, channels]'
        # "we need placeholders when we want to input some data into our networks"

        # get our train and eval data
        self.train_images, self.train_labels = train_data
        self.eval_images, self.eval_labels = eval_data
        self.test_images, self.test_labels = test_data

        self.images_placeholder = tf.placeholder(tf.float64, shape=[batch_size,IMAGE_INPUT_SIZE,
                                            IMAGE_INPUT_SIZE,CHANNELS],name='images_placeholder')
        self.labels_placeholder = tf.placeholder(tf.float64, shape=[batch_size,NUM_LABELS],
                                            name='labels_placeholder')
        self.conv1_weights = tf.Variable(initial_value=tf.truncated_normal([4,4,CHANNELS,16]),name='weights_conv_1')
        self.bias1_weights = tf.Variable(initial_value=tf.zeros([16]),name='bias_conv_1')
        self.conv2_weights = tf.Variable(initial_value=tf.truncated_normal([4,4,16,32]), name='weights_conv_2')
        self.bias2_weights = tf.Variable(initial_value=tf.zeros([32]),name='bias_conv_2')
        self.fc_1_weights = tf.Variable(initial_value=tf.truncated_normal(
                [IMAGE_INPUT_SIZE // 4 * IMAGE_INPUT_SIZE // 4 * 32, 512]), name='weights_fc_1')
        self.fc_1_biases = tf.Variable(initial_value=tf.zeros([512]), name='bias_fc_1')
        self.fc_2_weights = tf.Variable(initial_value=tf.truncated_normal([512,NUM_LABELS]), name='weights_fc_2')
        self.fc_2_biases = tf.Variable(initial_value=tf.zeros([NUM_LABELS]), name='bias_fc_2')

        # some other variable if needed
        self.saver = tf.train.Saver(max_to_keep=10)


    def create_model(self, learn_rate=3e-4, reg_factor=1e-4):
        'this will create the model'
        self.learn_rate = learn_rate
        self.reg_factor = reg_factor

        input_ = tf.cast(self.images_placeholder,tf.float32)
        "remember that 'SAME' will only retain the size of input tensor if its stride is set to [1,1,1,1]"
        features = tf.nn.conv2d(input_, self.conv1_weights, strides=[1,1,1,1], padding='SAME',
                                name='conv_1')
        features = tf.nn.relu(tf.nn.bias_add(features,self.bias1_weights,name='bias_1'),name='relu_1')
        features = tf.nn.max_pool(features, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool_1')
        features = tf.nn.conv2d(features, self.conv2_weights, strides=[1,1,1,1], padding='SAME', name='conv_2')
        features = tf.nn.relu(tf.nn.bias_add(features,self.bias2_weights,name='bias_2'),name='relu_2')
        features = tf.nn.max_pool(features, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool_2')
        feat_shape = features.get_shape()
        features = tf.reshape(features, shape=[feat_shape[0],feat_shape[1]*feat_shape[2]*feat_shape[3]],
                                name='full_conn_input')
        print('features', features.get_shape(), self.fc_1_weights.get_shape())
        # print('log: shape before first fully connected layer: {}'.format(feat_shape))
        features = tf.nn.bias_add(tf.matmul(features,self.fc_1_weights), self.fc_1_biases, name=None)
        features = tf.nn.relu(features, name='fc_1')
        # the following will be our final output layer, the logits
        features = tf.nn.dropout(features, keep_prob=0.8)
        self.logits = tf.nn.bias_add(tf.matmul(features,self.fc_2_weights), self.fc_2_biases, name='logits')
        # this is the output of the model, the logits

        # now we shall introduce the training loss function
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=self.logits))
        # and a bit of regularization
        self.regu = 0  # tf.add(tf.nn.l2_loss(self.fc_1_weights),tf.nn.l2_loss(self.fc_1_biases))
        # now we define the loss function
        self.loss += reg_factor * self.regu
        # now choose a learning algorithm, by default it is SGD with Adam optimizer
        self.optim = tf.train.AdamOptimizer(learn_rate, 0.9, name="Adam").minimize(self.loss)
        # these are our final predictions
        self.pred = tf.nn.softmax(self.logits, name='pred')


    def train(self, sess, model_folder='models', save_after=100, eval_after=100, summary_dir='summaries'):
        'this will train the model, and it will be called inside a session so no need to create another one here'
        # 'we shall also save the model in this function'

        # print('logits ', logits.get_shape())
        # print('pred ', pred.get_shape())
        # make a summary folder
        if tf.gfile.Exists(summary_dir):
            call('rm -r {}'.format(summary_dir), shell=True)
            print('log: removing older folder {}'.format(summary_dir))
            call('mkdir {}'.format(summary_dir), shell=True)
        else:
            call('mkdir {}'.format(summary_dir), shell=True)

        # we shall add the summaries here
        # acc_summary = tf.summary.scalar('batch accuracy', batch_accuracy)
        loss_summary = tf.summary.scalar('batch_loss', self.loss)
        # now merge them
        self.merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=sess.graph)
        # summary_writer.add_graph(graph=sess.graph, global_step=None, graph_def=sess.graph_def)
        # summary_writer.add_graph(sess.graph)

        # the following is our training loop
        set = {0}
        sess.run(tf.global_variables_initializer())
        for step in xrange(NUM_EPOCHS*NUM_TRAIN_IMAGES//TRAIN_BATCH_SIZE):
            random_batch = random.sample(range(0,NUM_TRAIN_IMAGES), TRAIN_BATCH_SIZE)
            for x in random_batch: set.add(x)
            train_images_batch = self.train_images[random_batch, ...]
            train_labels_batch = self.train_labels[random_batch, ...]

            # print(train_labels_batch.shape)
            # one_image = train_images_batch[0, ...]
            # print(np.argmax(train_labels_batch[0,:],0))
            # one_image = np.array(one_image,dtype=np.int8)
            # one_image = np.reshape(one_image, (28,28))
            # plot.imshow(one_image, cmap='gray')
            # plot.show()
            sess.run(self.optim,feed_dict={self.images_placeholder:train_images_batch,
                                    self.labels_placeholder:train_labels_batch})
            batch_loss, batch_pred, summ = sess.run([self.loss, self.pred, self.merged_summary],
                                                    feed_dict={self.images_placeholder:train_images_batch,
                                                               self.labels_placeholder:train_labels_batch})
            # print(batch_pred, train_labels_batch)
            # print(batch_pred.shape[1], np.sum(np.argmax(batch_pred,1) == np.argmax(train_labels_batch,1))/int(batch_pred.shape[0]))
            batch_accuracy = 100*int(np.sum(np.argmax(batch_pred,1) == np.argmax(train_labels_batch,1))) / int(batch_pred.shape[0])
            # print(batch_accuracy)
            if step % TRAIN_BATCH_SIZE == 0:
                print('log: epoch = {}, batch loss = {}, batch accuracy = {}%'.format(step*NUM_TEST_IMAGES//TRAIN_BATCH_SIZE,
                                                                                      batch_loss, batch_accuracy))

            if step % save_after == 0:
                self.eval_model(sess=sess)

            # save the model
            if step % save_after == 0:
                self.save_model(session=sess, model_folder=model_folder, step=step)
            # save summaries
            # loss_summary = tf.summary.scalar('batch_loss', batch_loss)
            summary_writer.add_summary(summ, step)
            # summary_writer.add_summary(batch_pred, step)
        summary_writer.close()
        print('log: total images seen in training: ',len(set))

    def over_all_train_acc(self, sess, model_path):
        'this returns the overall accuracy of the trained model on the whole of training set'
        # 'we shall begin by loading our final trained model'
        total = 0
        # self.saver.restore(sess=sess, save_path=model_path)
        # print('log: model {} restored!'.format(model_path))
        this = 0
        for step in xrange(0, NUM_TRAIN_IMAGES, TRAIN_BATCH_SIZE):
            # print(step, step+TRAIN_BATCH_SIZE, NUM_TRAIN_IMAGES//TRAIN_BATCH_SIZE)
            train_images_batch = self.train_images[step:step+TRAIN_BATCH_SIZE, ...]
            train_labels_batch = self.train_labels[step:step+TRAIN_BATCH_SIZE, ...]
            this+= TRAIN_BATCH_SIZE
            batch_pred = sess.run(self.pred, feed_dict={self.images_placeholder:train_images_batch,
                                                              self.labels_placeholder:train_labels_batch})
            batch_correct_pred = int(np.sum(np.argmax(batch_pred,axis=1) == np.argmax(train_labels_batch, axis=1)))
            total += batch_correct_pred
        # print(total / NUM_TRAIN_IMAGES)
        full_acc = total / this
        print('log: training score = correct / total: {}/{} = {}%'.format(total, this, 100*total/this))
        return 100*full_acc

    def eval_model(self, sess):
        # 'evaluate the model on evaluation set, sometimes during training'
        total = 0
        # self.saver.restore(sess=sess, save_path=model_path)
        # print('log: model {} restored!'.format(model_path))
        this = 0
        for step in xrange(0, NUM_EVAL_IMAGES, EVAL_BATCH_SIZE):
            # print(step, step+TRAIN_BATCH_SIZE, NUM_TRAIN_IMAGES//TRAIN_BATCH_SIZE)
            eval_images_batch = self.eval_images[step:step+EVAL_BATCH_SIZE, ...]
            eval_labels_batch = self.eval_labels[step:step+EVAL_BATCH_SIZE, ...]
            this += EVAL_BATCH_SIZE
            batch_pred = sess.run(self.pred, feed_dict={self.images_placeholder:eval_images_batch,
                                                              self.labels_placeholder:eval_labels_batch})
            batch_correct_pred = int(np.sum(np.argmax(batch_pred,axis=1) == np.argmax(eval_labels_batch, axis=1)))
            total += batch_correct_pred
        # print(total / NUM_TRAIN_IMAGES)
        full_acc = total / this
        print('log: evaluation score = correct / total: {}/{} = {}%'.format(total, this, 100*total/this))

    def test_model(self, sess):
        # 'test on the test set'
        total = 0
        # self.saver.restore(sess=sess, save_path=model_path)
        # print('log: model {} restored!'.format(model_path))
        this = 0
        for step in xrange(0, NUM_TEST_IMAGES, TEST_BATCH_SIZE):
            # print(step, step+TRAIN_BATCH_SIZE, NUM_TRAIN_IMAGES//TRAIN_BATCH_SIZE)
            test_images_batch = self.test_images[step:step+TEST_BATCH_SIZE, ...]
            test_labels_batch = self.test_labels[step:step+TEST_BATCH_SIZE, ...]
            this += TEST_BATCH_SIZE
            batch_pred = sess.run(self.pred, feed_dict={self.images_placeholder:test_images_batch,
                                                              self.labels_placeholder:test_labels_batch})
            batch_correct_pred = int(np.sum(np.argmax(batch_pred,axis=1) == np.argmax(test_labels_batch, axis=1)))
            total += batch_correct_pred
        # print(total / NUM_TRAIN_IMAGES)
        full_acc = total / this
        print('log: test score = correct / total: {}/{} = {}%'.format(total, this, 100*total/this))
        return

    def save_model(self, session, model_folder, step):
        'saves the model as a ckpt file'
        save_path = os.path.join(os.getcwd(), model_folder)
        if not tf.gfile.Exists(save_path):
            call('mkdir {}'.format(save_path), shell=True)
            print('log: made dir = {}'.format(save_path))
        model_name = '{}/model_{}.ckpt'.format(save_path,step)
        self.saver.save(sess=session, save_path=model_name)
        print('log: model saved as {}'.format(model_name))
        return None
