

import numpy as np
import pylab as pl

# function that finds possible windows
def find_possible_windows(image, distance, clearance_size, veiw_angle):
  """ finds the pixel position of window that meet clearance size
  Args:
    image, distance image numpy array of size (h, w)
    clearence_size, clearance window size (same units as those in distance image)
    veiw_angle, view angle of image
  Return:
    (posx
  """
  image_shape = image.shape
  for x in xrange(image_shape[0]):
    for y in xrange(image_shape[1]):
      max_x, min_x, max_y, min_y = needed_window_size(image_shape[0], image_shape[1], 3.14/2, 3.14/2, x, y, image[x,y], clearance_size)
      if np.min(image[min_x:max_x,min_y:max_y]) > distance:
        return max_x, min_x, max_y, min_y  

   
# function that determines the window clearence size
def needed_window_size(image_x, image_y, veiw_angle_x, veiw_angle_y, pos_x, pos_y, pos_distance, clearance_size):

  # distance from center of image
  center_pos_x = pos_x - image_x/2.0
  center_pos_y = pos_y - image_y/2.0

  # this is the angle from the middle of the screen to the far edge
  center_veiw_angle_x = veiw_angle_x/2.0
  center_veiw_angle_y = veiw_angle_y/2.0

  # distance from center of image
  center_image_x = image_x/2.0
  center_image_y = image_y/2.0

  # distance from view point to screen (should be same for both x and y)
  image_distance = center_image_x/np.tan(center_veiw_angle_x)

  # angle for pos
  center_angle_pos_x = np.arctan(center_pos_x / image_distance)
  center_angle_pos_y = np.arctan(center_pos_y / image_distance)

  # distance from center in real world
  real_center_pos_x = pos_distance * np.sin(center_angle_pos_x)
  real_center_pos_y = pos_distance * np.sin(center_angle_pos_y)

  # distance to center in real world (should be same for both x and y)
  real_center_pos_distance = pos_distance * np.cos(center_angle_pos_x) 

  # distance from center to lower and upper piece
  real_center_pos_x_minus = real_center_pos_x - clearance_size/2
  real_center_pos_x_plus = real_center_pos_x + clearance_size/2
  real_center_pos_y_minus = real_center_pos_y - clearance_size/2
  real_center_pos_y_plus = real_center_pos_y + clearance_size/2

  # angle from center to lower and upper piece
  center_angle_pos_x_minus = np.arctan(real_center_pos_x_minus / real_center_pos_distance)
  center_angle_pos_x_plus = np.arctan(real_center_pos_x_plus / real_center_pos_distance)
  center_angle_pos_y_minus = np.arctan(real_center_pos_y_minus / real_center_pos_distance)
  center_angle_pos_y_plus = np.arctan(real_center_pos_y_plus / real_center_pos_distance)

  # window box
  max_x = (np.tan(center_angle_pos_x_plus) * image_distance) + image_x/2.0
  min_x = (np.tan(center_angle_pos_x_minus) * image_distance) + image_x/2.0
  max_y = (np.tan(center_angle_pos_y_plus) * image_distance) + image_y/2.0
  min_y = (np.tan(center_angle_pos_y_minus) * image_distance) + image_y/2.0
 
  # bound to window
  min_x = int(min(max(min_x, 0), image_x))
  min_y = int(min(max(min_y, 0), image_y))
  max_x = int(min(max(max_x, 0), image_x))
  max_y = int(min(max(max_y, 0), image_y))

  return max_x, min_x, max_y, min_y

# pretend distance image
#distance_image = np.zeros((400,400)) + 1.0
# set a few chunks to have distance father away
#distance_image[30:70,30:70] = 4.0
#distance_image[200:300,200:300] = 3.0
#distance_image[100:200,30:70] = 2.0
distance_image = np.loadtxt("depth.data")
distance_image = distance_image.reshape(376, 672)
where_are_NaNs = np.isnan(distance_image)
distance_image[where_are_NaNs] = 0


max_x, min_x, max_y, min_y = find_possible_windows(distance_image, 10.0, 1.0, 3.14)

pl.imshow(distance_image)
pl.show()

distance_image[min_x:max_x, min_y:max_y] = 20.0

pl.imshow(distance_image)
pl.show()

print(max_x)
print(min_x)
print(max_y)
print(min_y)

