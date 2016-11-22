

import numpy as np

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

  return threshold_depth


# this calcs the threshold for a 400x400 image, view angle 2 rads, window size 31 (4 layers of 3x3max pool with stride 2), clearance size 1 meter
test_threshold = threshold_depth_calc(400, 400, 2.0, 2.0, 31, 1.0)
print("test threshold " + str(test_threshold))

# this mimics the thresholds needed for 5 3x3max pool with stride 2
print("This mimics the thresholds needed for each layer of a 3 by 3 max pool with stride 2.")
print("The values produced are the minimum distance away required for each pixel in the window.")
threshold_layer_1 = threshold_depth_calc(400, 400, 2.0, 2.0, 3, 1.0)
print("threshold layer 1 = " + str(threshold_layer_1) + " meters")
threshold_layer_2 = threshold_depth_calc(400, 400, 2.0, 2.0, 7, 1.0)
print("threshold layer 2 = " + str(threshold_layer_2) + " meters")
threshold_layer_3 = threshold_depth_calc(400, 400, 2.0, 2.0, 15, 1.0)
print("threshold layer 3 = " + str(threshold_layer_3) + " meters")
threshold_layer_4 = threshold_depth_calc(400, 400, 2.0, 2.0, 31, 1.0)
print("threshold layer 4 = " + str(threshold_layer_4) + " meters")
threshold_layer_5 = threshold_depth_calc(400, 400, 2.0, 2.0, 63, 1.0)
print("threshold layer 5 = " + str(threshold_layer_5) + " meters")
threshold_layer_6 = threshold_depth_calc(400, 400, 2.0, 2.0, 127, 1.0)
print("threshold layer 6 = " + str(threshold_layer_6) + " meters")
threshold_layer_7 = threshold_depth_calc(400, 400, 2.0, 2.0, 255, 1.0)
print("threshold layer 7 = " + str(threshold_layer_7) + " meters")


