""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import random

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from sklearn import preprocessing

# Training Parameters
from sklearn.cross_validation import train_test_split


class autoencoder():
    def __init__(self, num_input, batch_size = 64, num_hidden_1 = 128, num_hidden_2 = 20):
        self.num_input = num_input
        self.batch_size = batch_size
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.num_steps = 30000
        self.display_step = 1000
        self.read_data()
        self.__init_graph()

    def read_data(self):
        self.data = pd.read_csv("./data/yeast/data.txt", sep=" ", header=None)
        self.data = np.array(self.data.iloc[:, 1:])
        # scaler = MinMaxScaler()
        # X_train = scaler.fit_transform(data)
        self.dataset = preprocessing.minmax_scale(self.data, feature_range=(0, 1))
        self.X_train, self.X_test = train_test_split(self.dataset, test_size=0.5)


    def __init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):

            # tf Graph input (only pictures)
            self.X = tf.placeholder("float", [None, self.num_input])

            self.weights, self.biases = self._initialize_weights()
            # Building the encoder
            def encoder(x):
                # Encoder Hidden layer with sigmoid activation #1
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                               self.biases['encoder_b1']))
                # Encoder Hidden layer with sigmoid activation #2
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                               self.biases['encoder_b2']))
                return layer_2


            # Building the decoder
            def decoder(x):
                # Decoder Hidden layer with sigmoid activation #1
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                               self.biases['decoder_b1']))
                # Decoder Hidden layer with sigmoid activation #2
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                               self.biases['decoder_b2']))
                return layer_2

            # Construct model
            self.encoder_op = encoder(self.X)
            self.decoder_op = decoder(self.encoder_op)

            # Prediction
            y_pred = self.decoder_op
            # Targets (Labels) are the input data.
            y_true = self.X

            # Define loss and optimizer, minimize the squared error
            self.loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_input])),
        }
        return weights, biases

    def train(self):  # fit a dataset
        # Training
        for i in range(1, self.num_steps+1):
            # Prepare Data
            ind = random.sample(range(len(self.dataset)), self.batch_size)
            batch_xs = self.dataset[ind]
            # Get the next batch of MNIST data (only images are needed, not labels)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_xs})
            # Display logs per step
            if i % self.display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))

        # Testing
        # Encode and decode images from test set and visualize their reconstruction.
        # Encode and decode the digit image
        weights = {}
        weights['w1'] = self.sess.run(self.weights['encoder_h1'])
        weights['w2'] = self.sess.run(self.weights['encoder_h2'])
        weights['b1']= self.sess.run(self.biases['encoder_b1'])
        weights['b2'] = self.sess.run(self.biases['encoder_b2'])

        # pd.DataFrame(g).to_csv("./data/yeast/expression_embeddings.txt", index=True, columns=None,header=None)

        return weights