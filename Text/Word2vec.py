'''
Usage:

index2word = {0: "One", 1: "Two", 2: "Three", 3: "Four", 4: "Five", 5: "Six", 6: "Seven", 7: "Eight", 8: "Nine"}
word2index = {v:k for k, v in index2word.items()}
odd = np.random.choice(range(0,10,2),size = (2000,2))
even = np.random.choice(range(1,9,2),size = (2000,2))
result = np.concatenate([odd,even], axis = 0)
#change to int32 type
X = np.int32(result[:,0])
Y = np.int32(result[:,1].reshape(-1,1))

tf.reset_default_graph()
c = TextProcessor(vocabulary_size = len(word2index))
pr = c.predict()

ref_word = pr[word2index["One"]]
cosine_dists = np.dot(pr, ref_word)
ff = np.argsort(cosine_dists)[::-1][1:10]
for f in ff:
    print(index2word[f])
    print(cosine_dists[f])

'''

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.utils import shuffle

word2index = {}
index2word = {}
batch_size = 64
embedding_dimension = 5
negative_samples = 8
#vocabulary_size = len(index2word_map)

class TextProcessor:
    
    def __init__(self, vocabulary_size, log_dir = 'Test', batch = 4000, embedding_dimension = 5, negative_samples = 8):
        self.batch = batch
        self.embedding_dimension = embedding_dimension
        self.negative_samples = negative_samples
        self.vocabulary_size = vocabulary_size
        #directory for saving model and tensorboard stuff
        self.log_dir = log_dir
        #input
        self.X = tf.placeholder(tf.int32, shape=[self.batch])
        self.Y = tf.placeholder(tf.int32, shape=[self.batch, 1])
        # Embedding lookup table currently only implemented in CPU
        with tf.name_scope("embeddings"):
            self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_dimension],-1.0, 1.0), name='embedding')
            # This is essentialy a lookup table
            self.embed = tf.nn.embedding_lookup(self.embeddings,self.X)
        # Create variables for the NCE loss
        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_dimension], stddev=1.0 / math.sqrt(self.embedding_dimension)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights = self.nce_weights, biases = self.nce_biases, inputs = self.embed, labels = self.Y, 
                                                  num_sampled = self.negative_samples, num_classes = self.vocabulary_size))
        tf.summary.scalar("NCE_loss", self.loss)
        # Learning rate decay
        global_step = tf.Variable(0, trainable=False)
        self.learningRate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=1000, decay_rate=0.95, staircase=True)
        self.train_step = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.loss)
        self.merged = tf.summary.merge_all()
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        
    def save_metadata(self, k,v):
        with open(os.path.join(self.log_dir, 'metadata.tsv'), "w+") as metadata:
            metadata.write('Name\tClass\n')
            for k, v in index2word_map.items():
                metadata.write('%s\t%d\n' % (v, k))
    
    def fit(self, X,Y, epochs = 5001, every = 500):
        print('Starting')
        train_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())
        saver = tf.train.Saver()
        config = projector.ProjectorConfig()
        self.embedding = config.embeddings.add()
        self.embedding.tensor_name = self.embeddings.name
        # Link this tensor to its metadata file (e.g. labels).
        self.embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(train_writer, config)
        self.sess.run(self.init_op)
        print('First Epoch')
        for step in range(epochs):
            x, y = shuffle(X,Y)
            x_batch = x[:self.batch]
            y_batch = y[:self.batch]
            summary, _ = self.sess.run([self.merged, self.train_step], feed_dict={self.X: x_batch, self.Y: y_batch})
            train_writer.add_summary(summary, step)
            if step % every == 0:
                saver.save(self.sess, os.path.join(self.log_dir, "model"), step)
                loss_value = self.sess.run(self.loss, feed_dict = {self.X: x_batch, self.Y: y_batch})
                print("Loss at %d: %.5f" % (step, loss_value))

    def predict(self):              
        # Normalize embeddings before using
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
        normalized_embeddings = self.embeddings / norm
        normalized_embeddings_matrix = self.sess.run(normalized_embeddings)
        return normalized_embeddings_matrix
