'''
few changes to be able to use RNNs to binary classification
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler


CONNECTION = MongoClient('localhost')
db = CONNECTION.Bot
FxData = db.FxData

def normalize_maxmin(X):
    '''
    MaxMinNormalizer
    return: <numpy.ndarray>, with values between 0 and 1
    
    args:
        X: <numpy.ndarray>, features, labels whatever
    '''
    return MinMaxScaler().fit_transform(X)

def generate_x_y_fx(db, collection, field, minutes, delay = 0):
    '''
    generate_x_y_fx('Bot', 'FxData', 'Price', 1200)
    '''
    collection_length = CONNECTION[db][collection].count()
    length = minutes * 60
    random_skip = np.random.randint(0,collection_length - length)
    data = CONNECTION[db][collection].find({},{'_id':0,field:1}).skip(random_skip).limit(length * 2)
    clean_data = [item.get(field) for item in data]
    aggregated_data = list(np.mean(np.array(clean_data).reshape(-1, minutes), axis=1))
    if delay !=0:
        return aggregated_data[:minutes], aggregated_data[delay: delay + minutes]
    else:
        return aggregated_data[:minutes], aggregated_data[minutes:]

def generate_batch_fx(batch, db, collection, field, minutes, delay = 0):
    '''
    generate_batch_fx(1000, 'Bot', 'FxData', 'Price', 1200, 0)
    '''
    batch_x = []
    batch_y = []
    for item in range(batch):
        try:
            x, y = generate_x_y_fx(db, collection, field, minutes, delay)
            batch_x.append(x)
            batch_y.append(y)
            if item % 100 == 0:
                print(time.ctime(), str(item))
        except ValueError:
            pass
    return np.array(batch_x), np.array(batch_y)
        
        
def generate_x_y(batch_size = 50, sequence_length_x = 20, sequence_length_y = 20):
    """
    returns: tuple (X, Y)
        X is a sine and a cosine from 0.0*pi to 1.5*pi
        Y is a sine and a cosine from 1.5*pi to 3.0*pi
    Therefore, Y follows X. There is also a random offsetcommonly applied to X an Y.
    
    The returned arrays are of shape: (batch_size, sequence_length_ (x or y), 1)
    
    args:
        batch_size: <int>, size of generated batch
        sequence_length_x: <int>, size of x sequence, default 20
        sequence_length_y: <int>, size of y sequence, default 20
        
    Please note, with changing sequence_length ratios you will shorter array with zeros this can be useful if you want predict only next 5 values
    """
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        rand = np.random.random() * 2 * np.pi
        sig1 = np.sin(np.linspace(0.0 * np.pi + rand, 3.0 * np.pi + rand, sequence_length_x + sequence_length_y)).reshape(-1,1)
        x1 = sig1[:sequence_length_x]
        y1 = sig1[sequence_length_x:]
        if sequence_length_y < sequence_length_x:
            zeros = np.zeros((x1.shape[0] - y1.shape[0],1))
            y1 = np.concatenate((zeros, y1))
        elif sequence_length_y > sequence_length_x:
            zeros = np.zeros((y1.shape[0] - x1.shape[0],1))
            x1 = np.concatenate((zeros, x1))
        batch_x.append(x1)
        batch_y.append(y1)
    return np.array(batch_x), np.array(batch_y)

class RNNModel:
    '''
    basic implementation of RNN for predicting timeseries
    '''
    def __init__(self, learning_rate = 0.001, n_inputs = 1, n_outputs = 1, sequence_length = 20, n_neurons = 100):
        #parameters
        self.steps = 0
        self.sequence_length = sequence_length
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        #placeholders
        self.X = tf.placeholder(tf.float32, [None, self.sequence_length, self.n_inputs])
        self.Y = tf.placeholder(tf.float32, [None, self.sequence_length, self.n_outputs])
        #Layers
        OutputProjectionWrapper = tf.contrib.rnn.OutputProjectionWrapper
        BasicRNNCell = tf.nn.rnn_cell.BasicRNNCell
        ReLU = tf.nn.relu
        self.cell = OutputProjectionWrapper(BasicRNNCell (num_units = self.n_neurons, activation = ReLU), output_size = self.n_outputs)
        self.Y_hat, states = tf.nn.dynamic_rnn(self.cell, self.X, dtype=tf.float32)
        #loss, optimizer, R squared
        self.loss = tf.reduce_mean(tf.square(self.Y_hat - self.Y)) # MSE
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)
        # R squared
        self.total_error = tf.reduce_sum(tf.square(tf.subtract(self.Y, tf.reduce_mean(self.Y))))
        self.unexplained_error = tf.reduce_sum(tf.square(tf.subtract(self.Y_hat, tf.reduce_mean(self.Y))))
        self.R_squared = tf.subtract(1., tf.div(self.unexplained_error, self.total_error))
        #init and session
        self.init = tf.global_variables_initializer()
        #self.config = tf.ConfigProto(device_count = {'GPU': 0})
        self.sess = tf.Session()#(config = self.config)
        #saver
        self.saver = tf.train.Saver()
        #tensorboard stuff
        self.summary_loss = tf.summary.scalar('MSE', self.loss)
        self.summary_R_squared = tf.summary.scalar('R2', self.R_squared)

        
    def fit(self, X, Y, epochs = 1000, every = 100, log_path = 'RNNLog', batch_size = 500, init = True ):
        '''
        optimizes weights
        args:
          X: <numpy.array>, features 
          Y: <numpy.array>, labels
          epochs: <int>, number of epochs
          every: <int>, how often print message with Epoch and Loss values
          log_path: <string>, name of folder where to store data for TensorBoard
          batch_size: <int>, size of features array, size of features sample
          init: <boolean>, if second time running and you dont want initialize it again and use already calculated Variables
        '''
        if init:
            self.sess.run(self.init)
            self.steps = 0
        summary_writer = tf.summary.FileWriter(log_path, self.sess.graph)
        x_length = len(X)
        for i in range(self.steps, epochs + self.steps):
            self.steps += 1
            batch = np.random.randint(0, x_length - batch_size, 1)[0]
            x_batch = X[batch:batch + batch_size]
            y_batch = Y[batch:batch + batch_size]
            _, l, a = self.sess.run([self.train_step, self.loss, self.R_squared], feed_dict = {self.X: x_batch, self.Y: y_batch})
            loss, r_squared = self.sess.run([self.summary_loss, self.summary_R_squared], feed_dict = {self.X: x_batch, self.Y: y_batch})
            summary_writer.add_summary(loss, i)
            summary_writer.add_summary(r_squared, i)
            if i % every == 0:
                print('[ {} ] Epoch {} Loss: {:.7f} R squared:{:.3f}'.format(time.ctime(), i, l, a))
    
    def predict(self, X):
        '''
        return predicted values, 
        
        args:
          X: <numpy.array>, features
        '''
        Y_hat = self.sess.run(self.Y_hat, feed_dict = {self.X: X})
        return Y_hat
    
    def generate(self,X, length = 100):
        '''
        will generate array with length by adding second element from prediction to input X
        
        X needs to be in shape [1,self.sequence_length, self.n_outputs]
        
        it is using predicted values to generate predicted values 
        
        args:
            X: <numpy.array>,
            length: <int>, how many cycles to do 
        '''
        sequence = X
        for i in range(length):
            ind = sequence.shape[1] - self.sequence_length
            sequence_to_feed = np.array_split(sequence, [ind],axis = 1)[-1]
            Y_hat = self.sess.run(self.Y_hat, feed_dict = {self.X: sequence_to_feed})
            sequence = np.concatenate((sequence, Y_hat[0][self.n_outputs].reshape(1,1,1)), axis = 1)
        return sequence
    
    def generate_sequences(self,X, length = 2):
        '''
        will concatenate results to input X
        if sequence length is 20 and length 2 results will be 20 + 2 * 20
        '''
        sequence = X
        for i in range(length):
            ind = sequence.shape[1] - self.sequence_length
            sequence_to_feed = np.array_split(sequence, [ind],axis = 1)[-1]
            Y_hat = self.sess.run(self.Y_hat, feed_dict = {self.X: sequence_to_feed})
            shortened_Y_hat = np.array_split(Y_hat, [1],axis = 1)[-1]
            sequence = np.concatenate((sequence, shortened_Y_hat), axis = 1)
        return sequence
    
    def save(self, path):
        '''
        save model
        args:
            path: <string>, path with name where will be model stored
        path can be:
        path = './Model/RNN_{}'.format(time.ctime().replace(' ','_'))
        '''
        self.saver.save(self.sess, path)
        
    def close(self):
        '''
        close session
        '''
        self.sess.close()
        
    def close_and_reset(self):
        '''
        closes tensorflow session, clears the default graph stack and resets the global default graph
        '''
        self.sess.close()
        tf.reset_default_graph()
        
class MultiRNNModel(RNNModel):
	'''
	Stacked RNN cells
	'''
    def __init__(self, learning_rate = 0.0001, n_inputs = 1, n_outputs = 1, sequence_length = 20, n_neurons = 100, n_layers = 3):
        #parameters
        self.steps = 0
        self.sequence_length = sequence_length
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        #placeholders
        self.X = tf.placeholder(tf.float32, [None, self.sequence_length, self.n_inputs])
        self.Y = tf.placeholder(tf.float32, [None, self.sequence_length, self.n_outputs])
        #Layers
        OutputProjectionWrapper = tf.contrib.rnn.OutputProjectionWrapper
        ReLU = tf.nn.relu
        self.BasicCell = self.__set_RNN_cell_type()
        '''with tf.variable_scope("rnn1"):
            self.cell = BasicRNNCell (num_units = self.n_neurons, activation = ReLU)
            self.Y_hat_1, states = tf.nn.dynamic_rnn(self.cell, self.X, dtype=tf.float32)
        with tf.variable_scope("rnn2"):
            self.cell = BasicRNNCell (num_units = self.n_neurons, activation = ReLU)
            self.Y_hat_2, states = tf.nn.dynamic_rnn(self.cell, self.Y_hat_1, dtype=tf.float32)
        '''
        self.Y_generated = self.stack_RNNCell(self.n_layers - 1, self.n_neurons, self.X )
        with tf.variable_scope("rnnfinal"):
            self.cell = OutputProjectionWrapper(self.BasicCell (num_units = self.n_neurons, activation = ReLU), output_size = self.n_outputs)
            self.Y_hat, states = tf.nn.dynamic_rnn(self.cell, self.Y_generated, dtype=tf.float32)
        #self.cell = OutputProjectionWrapper(self.MultiCell, output_size = self.n_outputs)
        #loss, optimizer, R squared
        self.loss = tf.reduce_mean(tf.square(self.Y_hat - self.Y)) # MSE
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)
        # R squared
        self.total_error = tf.reduce_sum(tf.square(tf.subtract(self.Y, tf.reduce_mean(self.Y))))
        self.unexplained_error = tf.reduce_sum(tf.square(tf.subtract(self.Y_hat, tf.reduce_mean(self.Y))))
        self.R_squared = tf.subtract(1., tf.div(self.unexplained_error, self.total_error))
        #init and session
        self.init = tf.global_variables_initializer()
        self.config = tf.ConfigProto(device_count = {'GPU': 0})
        self.sess = tf.Session(config = self.config)
        #saver
        self.saver = tf.train.Saver()
        #tensorboard stuff
        self.summary_loss = tf.summary.scalar('MSE', self.loss)
        self.summary_R_squared = tf.summary.scalar('R2', self.R_squared)
        
    def __set_RNN_cell_type(self):
        '''
        overwrite return statement of this this function to use GRU or other LSTM cells
        '''
        return tf.nn.rnn_cell.BasicRNNCell
        
    def stack_RNNCell(self, layers, n_neurons, X):
        '''
        my solution how to stack RNN cells
        '''
        Y = X
        for i in range(layers):
            with tf.variable_scope('rnn{}'.format(i)):
                layer = self.BasicCell(num_units = self.n_neurons, activation = ReLU)
                Y, states = tf.nn.dynamic_rnn(layer, Y, dtype=tf.float32)
        return Y
    
class MultiGRUModel(RNNModel):
	'''
	Stacked GRU cells
	'''
    def __set_RNN_cell_type(self):
        '''
        overwrite return statement of this this function to use GRU or other LSTM cells
        '''
        return tf.nn.rnn_cell.GRUCell
