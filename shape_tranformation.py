import tensorflow as tf
import numpy as np
from ops import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class shapeZVector():

    def __init__(self):

        v = np.loadtxt('MyTestFile1.txt')
        image_matrix1 = np.reshape(v, (16, 16, 16)).astype(np.float32)
        self.volume = image_matrix1
        z,x,y = image_matrix1.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, -z, zdir='z', c='red')
        plt.savefig("demo.png")

        data_tf = tf.convert_to_tensor(image_matrix1, np.float32)

        sess = tf.InteractiveSession()
        print(data_tf.eval())
        print(tf.shape(data_tf))
        print(tf.rank(data_tf))


        images = tf.placeholder(tf.float32, [None,4096])
        dataset = tf.data.Dataset.from_tensor_slices((images))
        iterator = dataset.make_initializable_iterator()
        image_matrix1 = np.reshape(image_matrix1, (1,4096)).astype(np.float32)
        sess.run(iterator.initializer, feed_dict={images: image_matrix1})

        image_matrix = tf.reshape(images, [-1, 16, 16, 16,1])
        sess.close()
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
        samples = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        some_test = tf.constant(
            np.random.normal(loc=0.0, scale=1.0, size=(2, 2)).astype(np.float32))

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print("This is rank of Z tensor", tf.rank(guessed_z))
            print("This is Z ", guessed_z)
            print(type(guessed_z))
            guessed_z.eval(session = sess)

    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv3d(input_images, 1, 16, "d_h1"))  # 16x16x16x1 -> 8x8x8x16
            h2 = lrelu(conv3d(h1, 16, 32, "d_h2"))  # 8x8x8x16 -> 4x4x4x32
            h2_flat = tf.reshape(h2, [self.batchsize, 4 * 4 *4* 32])

            w_mean = dense(h2_flat, 4 * 4 * 4 * 32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 4 * 4 * 4 * 32, self.n_z, "w_stddev")

            return w_mean, w_stddev

data = np.asarray([[1,2,3],[4,5,6]])
print("This is shape of data", data)
n = tf.convert_to_tensor(data,np.float32)
sess = tf.InteractiveSession()
print(n.eval())
print(tf.shape(n))
print(tf.rank(n))
sess.close()
model = shapeZVector()
