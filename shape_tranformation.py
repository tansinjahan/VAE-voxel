import tensorflow as tf
import numpy as np

v = np.loadtxt('MyTestFile.txt')
print(v.shape)
#image_matrix = tf.reshape(v,[16, 16, 16])
image_matrix = np.reshape(v, (16,16,16))
print(image_matrix.shape)
sess = tf.InteractiveSession()
test = tf.constant(image_matrix)
sess.run(test)
sess.close()
print(image_matrix[:,:,0])