import tensorflow as tf
import numpy as np
from ops import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class shapeZVector():

    def __init__(self,v):

        image_matrix1 = np.reshape(v, (16, 16, 16)).astype(np.float32)
        self.volume = image_matrix1
        z,x,y = image_matrix1.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, -z, zdir='z', c='red')
        plt.savefig("demo.png")

        self.batchsize = 1
        self.n_z = 20

        tens = tf.constant(image_matrix1)
        mean, std = self.recognition(tf.reshape(tens, shape=[-1, 16, 16, 16, 1]))

        samples = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
        gussed_z = tf.add(mean, tf.multiply(std, samples))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(type(gussed_z))  # tensorflow.python.framework.ops.Tensor

            npArrr = sess.run(gussed_z)
            print(type(npArrr))  # numpy.ndarray
            print(npArrr)
            print("This is the shape of Z",npArrr.shape)

        return gussed_z

    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv3d(input_images, 1, 16, "d_h1"))  # 16x16x16x1 -> 8x8x8x16
            h2 = lrelu(conv3d(h1, 16, 32, "d_h2"))  # 8x8x8x16 -> 4x4x4x32
            h2_flat = tf.reshape(h2, [self.batchsize, 4 * 4 *4* 32])

            w_mean = dense(h2_flat, 4 * 4 * 4 * 32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 4 * 4 * 4 * 32, self.n_z, "w_stddev")

            return w_mean, w_stddev

def loadfile():
    temp = np.array([])
    for i in range(1, 11):
        v = np.loadtxt('MyTestFile' + str(i) + '.txt')
        model = shapeZVector(v)
        temp = np.append(temp,model)
    return temp

output = loadfile()
print("This is the Z vector for 10 shape", output)