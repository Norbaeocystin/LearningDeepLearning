%matplotlib inline
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time

def get_mnist():
    '''
    returns mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    '''
    mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def display_digit(num, x, y):
    '''
    display image from mnist data
    '''
    label = y[num].argmax(axis = 0)
    image = x[num].reshape([28,28])
    plt.tile('Example: {} Label: {}'.format(num, label))
    plt.imshow(image, cmap = plt.get_cmap('gray_r'))
    plt.show()
    
class CNNModel():
    '''
    Convolutional Neural Network
    '''
    
    def __init__(self, learning_rate = 0.0001, n_input = 784, n_hidden = 20, classes = 10, dropout = 0.85):
        self.learning_rate = learning_rate
        self.n_input = n_input
        self.classes = classes
        self.dropout = dropout
        # placeholders
        self.X = tf.placeholder(tf.float32,[None, self.n_input], name = 'X')
        self.Y = tf.placeholder(tf.float32, [None, self.classes], name = 'Y')
        self.x = tf.reshape(self.X, shape = [-1, 28, 28, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        #variables
        self.w1 = tf.Variable(tf.random_normal([5,5,1,32]))
        self.b1 = tf.Variable(tf.random_normal([32]))
        self.w2 = tf.Variable(tf.random_normal([5,5,32,64]))
        self.b2 = tf.Variable(tf.random_normal([64]))
        self.w3 = tf.Variable(tf.random_normal([7*7*64, 1024]))
        self.b3 = tf.Variable(tf.random_normal([1024]))
        self.w4 = tf.Variable(tf.random_normal([1024, self.classes]))
        self.b4 = tf.Variable(tf.random_normal([self.classes]))
        #layers
        self.conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.x, self.w1, strides = [1,1,1,1], padding = 'SAME'), self.b1))
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        self.conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.maxpool1, self.w2, strides = [1,1,1,1], padding = 'SAME'), self.b2))
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        self.fc1 = tf.reshape(self.maxpool2, [-1, self.w3.get_shape().as_list()[0]])
        self.fc2 = tf.add(tf.matmul(self.fc1, self.w3), self.b3)
        self.fc3 = tf.nn.relu(self.fc2)
        self.fc4 = tf.nn.dropout(self.fc3, self.dropout)
        # prediction 
        self.Y_hat = tf.add(tf.matmul(self.fc4, self.w4), self.b4)
        #loss and optimization
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.Y, logits = self.Y_hat, name = 'loss'))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.Y_hat, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # for tensorboard
        self.loss_scalar = tf.summary.scalar('cross-entropy', self.loss)
        self.accuracy_scalar = tf.summary.scalar('accuracy', self.accuracy)
        # inititialization and session stuff
        self.init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(device_count = {'GPU': 0})
        self.sess = tf.Session(config = config)
        #saver
        self.saver = tf.train.Saver()
        
    def fit(self, X, Y, epochs = 100, every = 10, folder = 'MNIST_CNN_log', batch_size = 500, init = True, learning_rate = 0.01, dropout = 0.85 ):
        '''
        optimizes weights
        args:
          X: <numpy.array>, features 
          Y: <numpy.array>, labels
          epochs: <int>, number of epochs
          every: <int>, how often print message with Epoch and Loss values
          folder: <string>, name of folder where to store data for TensorBoard
          batch_size: <int>, size of features array, size of features sample
          learning_rate: <float>, learning rate
          dropout: <float>, float between 0 and 1, how many neurons to keep 
        '''
        self.learning_rate = learning_rate
        self.dropout = dropout
        total = []
        if init:
            self.sess.run(self.init_op)
        summary_writer = tf.summary.FileWriter(folder, self.sess.graph)
        x_length = len(X)
        for i in range(epochs):
            batch = np.random.randint(0, x_length - batch_size, 1)[0]
            x_batch = X[batch:batch + batch_size]
            y_batch = Y[batch:batch + batch_size]
            _, l, a = self.sess.run([self.optimizer, self.loss, self.accuracy], feed_dict = {self.X: x_batch, self.Y: y_batch})
            loss, accuracy = self.sess.run([self.loss_scalar, self.accuracy_scalar], feed_dict = {self.X: x_batch, self.Y: y_batch})
            summary_writer.add_summary(loss, i)
            summary_writer.add_summary(accuracy, i)
            total.append(l)
            if i % every == 0:
                print('[ {} ] Epoch {} Loss: {:.7f} Accuracy:{:.3f}'.format(time.ctime(), i, l, a))
        return total
    
    def predict(self, X, before_fit = False):
        '''
        return predicted values, 
        
        args:
          X: <numpy.array>, features
          before_fit: <boolean>, change to True if you want to use before calling fit method
        '''
        if before_fit:
            self.sess.run(self.init_op)
        Y_hat = self.sess.run(self.correct_prediction, feed_dict = {self.X: X})
        return Y_hat
    
    def show(self, values):
        '''
        plot graph
        
        args:
          values: <list>, list or array of values to be plotted
        '''
        plt.plot(values)
        plt.show()
    
    def close_session(self):
        '''
        closes tensorflow session
        '''
        self.sess.close()
        return True
    
    def close_and_reset(self):
        '''
        closes tensorflow session, clears the default graph stack and resets the global default graph.
        '''
        self.sess.close()
        tf.reset_default_graph()
        return True
    
    def save(path):
        '''
        saves variables.
        
        args:
          path: <string>, path where will model saved
        '''
        self.saver.save(self.sess, path)
        
    def compare(X, Y, learning_rate_1 = 0.01, learning_rate_2 = 0.001, epochs_1 = 500, epochs_2 = 500, every_1 = 10, 
                every_2 = 10, batch_size_1 = 500, batch_size_2 = 500):
        '''
        will compare different settings
        '''
        pass
