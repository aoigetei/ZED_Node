
import time

import numpy as np
import tensorflow as tf
import pylab as pl

def tensor_to_1_0(tensor, threshold):
  tensor = tensor + threshold
  tensor = tf.sign(tensor)
  tensor = (tensor - 1.0)/(-2.0)
  return tensor

# function to determine the depth threshold given the max pool window size and desired clearence height
def threshold_depth_calc(image_x, image_y, veiw_angle_x, veiw_angle_y, max_pool_window_size, clearance_size):

  # this is the angle from the middle of the screen to the far edge
  center_veiw_angle_x = veiw_angle_x/2.0
  center_veiw_angle_y = veiw_angle_y/2.0

  # distance from center of image
  center_image_x = image_x/2.0
  center_image_y = image_y/2.0

  # distance from view point to screen (should be same for both x and y if there is no distortion in the image)
  image_distance = center_image_x/np.tan(center_veiw_angle_x)

  # ratio of pixel size to real world size
  ratio = max_pool_window_size / clearance_size

  # calc the threshold depth from the ratio of pixel size to real world size
  threshold_depth = image_distance / ratio

  print(threshold_depth)
  return threshold_depth

def find_path():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():

    # input depth image
    x = tf.placeholder(tf.float32, [1, 376, 672, 1])

    # first reduce the image my a factor of 8
    depth_conv_1 = tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], 'VALID') 
    depth_conv_2 = tf.nn.max_pool(depth_conv_1, [1,3,3,1], [1,2,2,1], 'VALID') 
    depth_conv_3 = tf.nn.max_pool(depth_conv_2, [1,3,3,1], [1,2,2,1], 'VALID') 
    depth_conv_4 = tf.nn.max_pool(depth_conv_3, [1,3,3,1], [1,2,2,1], 'VALID') 

    # parameters for window
    shape_x = int(depth_conv_4.get_shape()[1])
    shape_y = int(depth_conv_4.get_shape()[2])
    shape = (shape_x, shape_y)
    view_angle_x = 2.0 
    view_angle_y = 2.0 
    clearence_size= 1.0

    # do ever expanding boxes (calc max pool and convert to 0 or 1 if it meets clearence size)
    threshold_layer_4 = threshold_depth_calc(shape[0], shape[1], view_angle_x, view_angle_y, 1, clearence_size)
    depth_conv_4_1_0 = tensor_to_1_0(depth_conv_4, threshold_layer_4)

    depth_conv_5 = tf.nn.max_pool(depth_conv_4, [1,3,3,1], [1,1,1,1], 'SAME')   # frame size 3
    threshold_layer_5 = threshold_depth_calc(shape[0], shape[1], view_angle_x, view_angle_y, 3, clearence_size)
    depth_conv_5_1_0 = tensor_to_1_0(depth_conv_5, threshold_layer_5)

    depth_conv_6 = tf.nn.max_pool(depth_conv_5, [1,3,3,1], [1,1,1,1], 'SAME')   # frame size 7
    threshold_layer_6 = threshold_depth_calc(shape[0], shape[1], view_angle_x, view_angle_y, 7, clearence_size)
    depth_conv_6_1_0 = tensor_to_1_0(depth_conv_6, threshold_layer_6)

    depth_conv_7 = tf.nn.max_pool(depth_conv_6, [1,3,3,1], [1,1,1,1], 'SAME')   # frame size 9
    threshold_layer_7 = threshold_depth_calc(shape[0], shape[1], view_angle_x, view_angle_y, 9, clearence_size)
    depth_conv_7_1_0 = tensor_to_1_0(depth_conv_7, threshold_layer_7)

    depth_conv_8 = tf.nn.max_pool(depth_conv_7, [1,3,3,1], [1,1,1,1], 'SAME')   # frame size 11 
    threshold_layer_8 = threshold_depth_calc(shape[0], shape[1], view_angle_x, view_angle_y, 11, clearence_size)
    depth_conv_8_1_0 = tensor_to_1_0(depth_conv_8, threshold_layer_8)

    depth_conv_9 = tf.nn.max_pool(depth_conv_8, [1,3,3,1], [1,1,1,1], 'SAME')   # frame size 13
    threshold_layer_9 = threshold_depth_calc(shape[0], shape[1], view_angle_x, view_angle_y, 13, clearence_size)
    depth_conv_9_1_0 = tensor_to_1_0(depth_conv_9, threshold_layer_9)

    depth_conv_10 = tf.nn.max_pool(depth_conv_9, [1,3,3,1], [1,1,1,1], 'SAME')  # frame size 15
    threshold_layer_10 = threshold_depth_calc(shape[0], shape[1], view_angle_x, view_angle_y, 15, clearence_size)
    depth_conv_10_1_0 = tensor_to_1_0(depth_conv_10, threshold_layer_10)

    depth_conv_11 = tf.nn.max_pool(depth_conv_10, [1,3,3,1], [1,1,1,1], 'SAME') # frame size 17
    threshold_layer_11 = threshold_depth_calc(shape[0], shape[1], view_angle_x, view_angle_y, 17, clearence_size)
    depth_conv_11_1_0 = tensor_to_1_0(depth_conv_11, threshold_layer_11)

    depth_conv_12 = tf.nn.max_pool(depth_conv_11, [1,3,3,1], [1,1,1,1], 'SAME') # frame size 19
    threshold_layer_12 = threshold_depth_calc(shape[0], shape[1], view_angle_x, view_angle_y, 19, clearence_size)
    depth_conv_12_1_0 = tensor_to_1_0(depth_conv_12, threshold_layer_12)

    depth_conv_13 = tf.nn.max_pool(depth_conv_12, [1,3,3,1], [1,1,1,1], 'SAME') # frame size 21
    threshold_layer_13 = threshold_depth_calc(shape[0], shape[1], view_angle_x, view_angle_y, 21, clearence_size)
    depth_conv_13_1_0 = tensor_to_1_0(depth_conv_13, threshold_layer_13)

    # kill of non viable paths by looking at the layers with te larges frame size first
    depth_conv_12_1_0 = depth_conv_12_1_0 * depth_conv_13_1_0
    depth_conv_11_1_0 = depth_conv_11_1_0 * depth_conv_12_1_0
    depth_conv_10_1_0 = depth_conv_10_1_0 * depth_conv_11_1_0
    depth_conv_9_1_0 = depth_conv_9_1_0 * depth_conv_10_1_0
    depth_conv_8_1_0 = depth_conv_8_1_0 * depth_conv_9_1_0
    depth_conv_7_1_0 = depth_conv_7_1_0 * depth_conv_8_1_0
    depth_conv_6_1_0 = depth_conv_6_1_0 * depth_conv_7_1_0
    depth_conv_5_1_0 = depth_conv_5_1_0 * depth_conv_6_1_0
    depth_conv_4_1_0 = depth_conv_4_1_0 * depth_conv_5_1_0

    # determine maximum depth for each path
    depth_max = tf.maximum(threshold_layer_12*depth_conv_12_1_0, threshold_layer_13*depth_conv_13_1_0)
    depth_max = tf.maximum(threshold_layer_11*depth_conv_11_1_0, depth_max)
    depth_max = tf.maximum(threshold_layer_10*depth_conv_10_1_0, depth_max)
    depth_max = tf.maximum(threshold_layer_9*depth_conv_9_1_0, depth_max)
    depth_max = tf.maximum(threshold_layer_8*depth_conv_8_1_0, depth_max)
    depth_max = tf.maximum(threshold_layer_7*depth_conv_7_1_0, depth_max)
    depth_max = tf.maximum(threshold_layer_6*depth_conv_6_1_0, depth_max)
    depth_max = tf.maximum(threshold_layer_5*depth_conv_5_1_0, depth_max)
    depth_max = tf.maximum(threshold_layer_4*depth_conv_4_1_0, depth_max)

    # Start running operations on the Graph.
    sess = tf.Session()

    # load in a test frame
    depth_image = np.loadtxt("depth.data")
    depth_image = -depth_image.reshape(1, 376, 672, 1)
    where_are_NaNs = np.isnan(depth_image) # I think there is a faster way to do this
    depth_image[where_are_NaNs] = -1.0

    # eval depths
    t = time.time() # to time it
    depth_conv_max_g = sess.run([depth_max],feed_dict={x:depth_image})[0]
    depth_conv_max_g = sess.run([depth_max],feed_dict={x:depth_image})[0]
    depth_conv_max_g = sess.run([depth_max],feed_dict={x:depth_image})[0]
    depth_conv_max_g = sess.run([depth_max],feed_dict={x:depth_image})[0]
    depth_conv_max_g = sess.run([depth_max],feed_dict={x:depth_image})[0]
    #depth_conv_max_g = sess.run([depth_conv_13_1_0],feed_dict={x:depth_image})[0]
    elapsed = time.time() - t
    print("time elapsed " + str(elapsed/5))
    
    # display max depth map 
    pl.figure(0) 
    pl.imshow(depth_image[0,:,:,0])
    pl.figure(1)
    pl.imshow(depth_conv_max_g[0,:,:,0])
    pl.show()
    print(depth_conv_max_g[0,:,:,0])
    pl.show()


def main(argv=None):  # pylint: disable=unused-argument
  find_path()

if __name__ == '__main__':
  tf.app.run()


