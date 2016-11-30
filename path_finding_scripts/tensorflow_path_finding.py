
import time

import numpy as np
import tensorflow as tf
import pylab as pl

def find_path():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():

    # input depth image
    x = tf.placeholder(tf.float32, [1, 376, 672, 1])
    x = -x

    # first reduce the image my a factor of 8
    depth_conv_1 = tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], 'VALID') 
    depth_conv_2 = tf.nn.max_pool(depth_conv_1, [1,3,3,1], [1,2,2,1], 'VALID') 
    depth_conv_3 = tf.nn.max_pool(depth_conv_2, [1,3,3,1], [1,2,2,1], 'VALID') 
    depth_conv_4 = tf.nn.max_pool(depth_conv_3, [1,3,3,1], [1,2,2,1], 'VALID') 
    # do ever expanding boxes
    depth_conv_4 = tf.nn.max_pool(depth_conv_4, [1,3,3,1], [1,1,1,1], 'SAME') 
    depth_conv_5 = tf.nn.max_pool(depth_conv_4, [1,3,3,1], [1,1,1,1], 'SAME') 
    depth_conv_6 = tf.nn.max_pool(depth_conv_5, [1,3,3,1], [1,1,1,1], 'SAME') 
    depth_conv_7 = tf.nn.max_pool(depth_conv_6, [1,3,3,1], [1,1,1,1], 'SAME') 
    depth_conv_8 = tf.nn.max_pool(depth_conv_7, [1,3,3,1], [1,1,1,1], 'SAME') 
    depth_conv_9 = tf.nn.max_pool(depth_conv_8, [1,3,3,1], [1,1,1,1], 'SAME')
    depth_conv_10 = tf.nn.max_pool(depth_conv_9, [1,3,3,1], [1,1,1,1], 'SAME') 
    depth_conv_11 = tf.nn.max_pool(depth_conv_10, [1,3,3,1], [1,1,1,1], 'SAME')
    # add them all up. The one with the biggest depth is the best path
    depth_conv_max = depth_conv_4 + depth_conv_5 + depth_conv_6 + depth_conv_7 + depth_conv_8 + depth_conv_9 + depth_conv_10 + depth_conv_11

    # Start running operations on the Graph.
    sess = tf.Session()

    # load in a test frame
    depth_image = np.loadtxt("depth.data")
    depth_image = -depth_image.reshape(1, 376, 672, 1)
    where_are_NaNs = np.isnan(depth_image) # I think there is a faster way to do this
    depth_image[where_are_NaNs] = 0

    # eval depths
    t = time.time() # to time it
    depth_conv_max_g = sess.run([depth_conv_max],feed_dict={x:depth_image})[0]
    elapsed = time.time() - t
    print("time elapsed " + str(elapsed))
    
    # display max depth map 
    pl.imshow(depth_image[0,:,:,0])
    pl.show()
    print(depth_conv_max_g[0,:,:,0])
    pl.imshow(depth_conv_max_g[0,:,:,0])
    pl.show()


def main(argv=None):  # pylint: disable=unused-argument
  find_path()

if __name__ == '__main__':
  tf.app.run()


