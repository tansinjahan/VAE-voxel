import tensorflow as tf
import numpy as np
from ops import *

class shapeZVector():

    def __init__(self):

        v = np.loadtxt('MyTestFile1.txt')
        self.volume = v
        print(v.shape)
        self.images = tf.placeholder(tf.float32, [None, 4096])
        image_matrix = tf.reshape(self.images, [-1, 16, 16, 16,1])
        self.batchsize = 1
        self.n_z = 20
        '''image_matrix = np.reshape(v, (16, 16, 16)).astype(np.float32)
        print(image_matrix.shape)
        sess = tf.InteractiveSession()
        test = tf.constant(image_matrix)
        sess.run(test)
        sess.close()
        print(image_matrix[:, :, 6])
        print(image_matrix[:, :, 4])
        print(image_matrix[:, :, 15])
        print(image_matrix[:, :, 14])'''

        z_mean, z_stddev = self.recognition(image_matrix)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv3d(input_images, 1, 16, "d_h1"))  # 16 -> 14x14x16
            h2 = lrelu(conv3d(h1, 16, 32, "d_h2"))  # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2, [self.batchsize, 7 * 7 * 32])

            w_mean = dense(h2_flat, 7 * 7 * 32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 7 * 7 * 32, self.n_z, "w_stddev")

            return w_mean, w_stddev

model = shapeZVector()

