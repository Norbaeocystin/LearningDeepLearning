"""
Implementation of the "A Neural Algorithm of Artistic Style" in Python using
TensorFlow.

use as:

vgg = VGGModel(path_content_image = 'data/My.jpg', path_style_image = 'data/Psychedelics.jpg')
vgg.fit(epochs = 1000, name = 'LSD')

"""
import imageio
import os
import sys
import numpy as np
from PIL import Image
import scipy.io
import scipy.misc
import skimage.transform
import tensorflow as tf

import logging


logging.basicConfig(level=logging.DEBUG, format = '%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

class VGGModel:
    
    def __init__(self, path_content_image = 'data/Marilyn_Monroe_in_1952.jpg', path_style_image = 'data/VanGogh-starry_night.jpg', 
                 noise_ratio = 0.6, beta = 5, alpha = 100, vgg_model = 'data/imagenet-vgg-verydeep-19.mat'):
        self.content_image = self.load_image(path_content_image)
        self.style_image = self.load_style_image(path_style_image)
        self.noise_ratio = noise_ratio
        self.alpha = alpha 
        self.beta = beta
        self.vgg_model = vgg_model
        
        
    def load_image(self, path):
        '''
        load image, add dimension
        '''
        image = imageio.imread(path)
        image = np.reshape(image, ((1, *image.shape)))
        image = image - MEAN_VALUES
        return image
    
    def load_style_image(self, path):
        '''
        loads style image and resized it to size of content image
        here it is little complicated because of rotation after resizing of image
        '''
        img = Image.open(path)
        _, h, w, _ = self.content_image.shape
        img = img.resize([h, w])
        image  = np.asarray(img)
        if image.shape != self.content_image.shape[1:]:
            # maybe only np.transpose(image,[1,0,2]) is enough
            image = np.transpose(np.transpose(image, (2, 0, 1)))
        image = image.reshape((self.content_image.shape))
        return image - MEAN_VALUES

    def generate_noise_image(self, content_image):
        """
        Returns a noise image intermixed with the content image at a certain ratio.
        """
        noise_image = np.random.uniform(-20, 20, content_image.shape).astype('float32')
        # White noise image from the content representation. Take a weighted average
        # of the values
        input_image = noise_image * self.noise_ratio + content_image * (1 - self.noise_ratio)
        return input_image

    def save_image(self, path, image):
        '''
        save image 
        '''
        image = image + MEAN_VALUES
        image = image[0]
        image = np.clip(image, 0, 255).astype('uint8')
        imageio.imwrite(path, image)

    def load_vgg_model(self, path):
        """
        Returns a model for the purpose of 'painting' the picture.
        Takes only the convolution layer weights and wrap using the TensorFlow
        Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
        the paper indicates that using AveragePooling yields better results.
        The last few fully connected layers are not used.
        """
        vgg = scipy.io.loadmat(path)

        vgg_layers = vgg['layers']
        def _weights(layer, expected_layer_name):
            """
            Return the weights and bias from the VGG model for a given layer.
            """
            W = vgg_layers[0][layer][0][0][0][0][0]
            b = vgg_layers[0][layer][0][0][0][0][1]
            layer_name = vgg_layers[0][layer][0][0][-2]
            assert layer_name == expected_layer_name
            return W, b

        def _relu(conv2d_layer):
            """
            Return the RELU function wrapped over a TensorFlow layer. Expects a
            Conv2d layer input.
            """
            return tf.nn.relu(conv2d_layer)

        def _conv2d(prev_layer, layer, layer_name):
            """
            Return the Conv2D layer using the weights, biases from the VGG
            model at 'layer'.
            """
            W, b = _weights(layer, layer_name)
            W = tf.constant(W)
            b = tf.constant(np.reshape(b, (b.size)))
            return tf.nn.conv2d(
                prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

        def _conv2d_relu(prev_layer, layer, layer_name):
            """
            Return the Conv2D + RELU layer using the weights, biases from the VGG
            model at 'layer'.
            """
            return _relu(_conv2d(prev_layer, layer, layer_name))

        def _avgpool(prev_layer):
            """
            Return the AveragePooling layer.
            """
            return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Constructs the graph model.
        graph = {}
        graph['input']   = tf.Variable(np.zeros(self.content_image.shape), dtype = 'float32')
        graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
        graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = _avgpool(graph['conv1_2'])
        graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = _avgpool(graph['conv2_2'])
        graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = _avgpool(graph['conv3_4'])
        graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = _avgpool(graph['conv4_4'])
        graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = _avgpool(graph['conv5_4'])
        return graph

    def content_loss_func(self, sess, model):
        """
        Content loss function as defined in the paper.
        """
        def _content_loss(p, x):
            N = p.shape[3]
            M = p.shape[1] * p.shape[2]
            return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
        return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

    def style_loss_func(self, sess, model):
        """
        Style loss function as defined in the paper.
        """
        def _gram_matrix(F, N, M):
            """
            The gram matrix G.
            """
            Ft = tf.reshape(F, (M, N))
            return tf.matmul(tf.transpose(Ft), Ft)

        def _style_loss(a, x):
            """
            The style loss calculation.
            """
            N = a.shape[3]
            M = a.shape[1] * a.shape[2]
            A = _gram_matrix(a, N, M)
            G = _gram_matrix(x, N, M)
            result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
            return result

        # Layers to use. We will use these layers as advised in the paper.
        # To have softer features, increase the weight of the higher layers
        # (conv5_1) and decrease the weight of the lower layers (conv1_1).
        # To have harder features, decrease the weight of the higher layers
        # (conv5_1) and increase the weight of the lower layers (conv1_1).
        layers = [
            ('conv1_1', 0.5),
            ('conv2_1', 1.0),
            ('conv3_1', 1.5),
            ('conv4_1', 3.0),
            ('conv5_1', 4.0),
        ]

        E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in layers]
        W = [w for _, w in layers]
        loss = sum([W[l] * E[l] for l in range(len(layers))])
        return loss


    def fit(self, output = 'output/', epochs = 4000, save_every = 100, name = ''):
        '''
        train model and save images generated
        '''
        with tf.Session(config=tf.ConfigProto(log_device_placement = True, device_count = {'GPU': 0})) as sess:
            # Load the images.
            # Load the model.
            logger.debug('Starting')
            model = self.load_vgg_model(self.vgg_model)
            logger.debug('Model loaded')
            input_image = self.generate_noise_image(self.content_image)
            logger.debug('Image noised')
            sess.run(tf.initialize_all_variables())
            logger.debug('Initialization done')
            sess.run(model['input'].assign(self.content_image))
            logger.debug('Content image assigned')
            content_loss = self.content_loss_func(sess, model)
            sess.run(model['input'].assign(self.style_image))
            logger.debug('Style image assigned')
            style_loss = self.style_loss_func(sess, model)
            total_loss = self.beta * content_loss + self.alpha * style_loss
            logger.debug('Total loss calculated')
            optimizer = tf.train.AdamOptimizer(2.0)
            train_step = optimizer.minimize(total_loss)
            sess.run(tf.initialize_all_variables())
            sess.run(model['input'].assign(input_image))
            logger.debug('Iterating started')
            for it in range(epochs):
                sess.run(train_step)
                if it % save_every == 0:
                    # Print every 100 iteration.
                    mixed_image = sess.run(model['input'])
                    logger.debug('Epoch: {}'.format(it))
                    logger.debug('sum : {}'.format(sess.run(tf.reduce_sum(mixed_image))))
                    logger.debug('cost: {}'.format(sess.run(total_loss)))
                    if not os.path.exists(output):
                        os.mkdir(output)
                    filename = 'output/%s_%d.png' % (name,it)
                    self.save_image(filename, mixed_image)
                if it == epochs - 1 and it % save_every != 0:
                    if not os.path.exists(output):
                            os.mkdir(output)
                    filename = 'output/%s_%s.png' % (name, 'Final')
                    self.save_image(filename, mixed_image)
