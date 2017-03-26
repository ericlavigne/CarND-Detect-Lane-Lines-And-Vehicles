import numpy as np
import cv2
from glob import glob

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.regularizers import l2

from moviepy.editor import VideoFileClip

import tensorflow as tf

def calibrate_chessboard():
  objp = np.zeros((6*9,3), np.float32)
  objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

  calibration_fnames = glob('camera_cal/calibration*.jpg')
  
  calibration_images = []
  objpoints = []
  imgpoints = []
  
  for fname in calibration_fnames:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    calibration_images.append(gray)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
      objpoints.append(objp)
      imgpoints.append(corners)
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                     calibration_images[0].shape[::-1], None, None)
  calibration = [mtx,dist]
  return calibration

def transform_image_files(transformation, src_pattern, dst_dir):
  src_fpaths = glob(src_pattern)
  for src_fpath in src_fpaths:
    img = cv2.imread(src_fpath)
    dst_img = transformation(img)
    fname = src_fpath.split('/')[-1]
    dst_fpath = dst_dir + '/' + fname
    cv2.imwrite(dst_fpath,dst_img)

def undistort(img, calibration):
  return cv2.undistort(img, calibration[0], calibration[1], None, calibration[0])

def undistort_files(calibration, src_pattern, dst_dir):
  transform_image_files((lambda x: undistort(x, calibration)), src_pattern, dst_dir)

def read_training_data_paths():
  """Returns {'x': [path1, path2, ...], 'y': [path1, path2, ...]}"""
  x = glob('training/*_x.png')
  y = glob('training/*_y.png')
  x.sort()
  y.sort()
  assert (len(x) == len(y)), "x and y files don't match"
  return {'x': x, 'y': y}

def read_training_y_file(fpath):
  """Read y file and convert to y format: two channels representing lane line and not lane line"""
  img = cv2.imread(fpath)
  img = crop_scale_white_balance(img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  normalized = np.zeros_like(img)
  normalized[img > 0] = 1
  return np.stack([normalized], axis=-1)

original_max_x = 1280
original_max_y = 720
crop_min_x = 200
crop_max_x = 1080
crop_min_y = 420
crop_max_y = 666
scale_factor=2

def crop_scale_white_balance(img):
  img = img[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
  img = cv2.resize(img, None, fx=(1.0/scale_factor), fy=(1.0/scale_factor),
                   interpolation=cv2.INTER_AREA)
  low = np.amin(img)
  high = np.amax(img)
  img = (((img - low + 1.0) * 252.0 / (high - low)) - 0.5).astype(np.uint8)
  return img

def uncrop_scale(img):
  img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
  if len(img.shape) == 2:
    img = cv2.merge((img,img,img))
  target_shape = (original_max_y,original_max_x, 3)
  frame = np.zeros(target_shape, dtype="uint8")
  frame[crop_min_y:crop_max_y, crop_min_x:crop_max_x, 0:3] = img
  img = frame
  return img

def preprocess_input_image(img):
  """Normalize to [-0.5,0.5] based on lightest and darkest pixel across all channels"""
  img = crop_scale_white_balance(img)
  img = cv2.GaussianBlur(img, (3,3), 0)
  return ((img / 253.0) - 0.5).astype(np.float32)

def read_training_data():
  """Returns tuple of input matrix and output matrix (X,y)"""
  paths = read_training_data_paths()
  X = []
  for x in paths['x']:
    X.append(preprocess_input_image(cv2.imread(x)))
  Y = []
  for y in paths['y']:
    Y.append(read_training_y_file(y))
  return {'x': np.stack(X), 'y': np.stack(Y)}

def lane_weighted_crossentropy(y_true, y_pred):
  """10x higher weight on prediction of lane markings vs background"""
  return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 10.0)

def compile_model(model):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer='adam',
                loss=lane_weighted_crossentropy,
                metrics=['binary_accuracy', 'binary_crossentropy'])

tf_pos_tanh_offset = tf.constant(0.5)
tf_pos_tanh_scale = tf.constant(0.45)

def tanh_zero_to_one(x):
  """Actually [0.05, 0.95] to avoid divide by zero errors"""
  return (tf.tanh(x) * tf_pos_tanh_scale) + tf_pos_tanh_offset

def create_model():
  """Create neural network model, defining layer architecture."""
  model = Sequential()
  # Convolution2D(output_depth, convolution height, convolution_width, ...)
  model.add(Convolution2D(20, 5, 5, border_mode='same',
            input_shape=(int((crop_max_y - crop_min_y) / scale_factor),
                         int((crop_max_x - crop_min_x) / scale_factor),
                         3)))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Convolution2D(30, 5, 5, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Convolution2D(30, 5, 5, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Convolution2D(30, 5, 5, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Convolution2D(20, 5, 5, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Convolution2D(10, 5, 5, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Convolution2D(1, 5, 5, border_mode='same', W_regularizer=l2(0.01), activation=tanh_zero_to_one))

  compile_model(model)
  return model

def train_model(model, validation_percentage=None, epochs=100):
  """Train the model. With so few examples, I usually prefer
     to use all examples for training. Setting aside some
     examples for validation is supported but not recommended."""
  data = read_training_data()
  if validation_percentage:
    return model.fit(data['x'], data['y'], nb_epoch=epochs, validation_split = validation_percentage / 100.0)
  else:
    return model.fit(data['x'], data['y'], nb_epoch=epochs)

def image_to_lane_markings(img, model):
  model_input = preprocess_input_image(img)[None, :, :, :]
  model_output = model.predict(model_input, batch_size=1)[0]
  lane_line_odds = cv2.split(model_output)[0]

  x_center = int(lane_line_odds.shape[1] / 2)

  threshold = 0.35
  #threshold = min(0.5, np.amax(lane_line_odds[:,:x_center]) - 0.1, np.amax(lane_line_odds[:,x_center:]) - 0.1)
  result = np.zeros_like(lane_line_odds)
  result[lane_line_odds > threshold] = 254
  
  #left_side = lane_line_odds[:,:x_center]
  #right_side = lane_line_odds[:,x_center:]
  #left_max_odds = np.amax(left_side)
  #right_max_odds = np.amax(right_side)
  #left_threshold = min(0.5, left_max_odds * 0.8)
  #right_threshold = min(0.5, right_max_odds * 0.8)
  #left_result = np.zeros_like(left_side)
  #right_result = np.zeros_like(right_side)
  #left_result[left_side > left_threshold] = 254
  #right_result[right_side > right_threshold] = 254
  #result = np.concatenate([left_side,right_side], axis=1)
  #print("odds:" + str(lane_line_odds.shape) + " left:" + str(left_result.shape) + " right:" + str(right_result.shape) + " result:" + str(result.shape))
  
  result = uncrop_scale(result)
  return result

# These parameters control both the size and vertical scaling of the image.
# I chose delta_x based on the width in pixels of the lane lines at the
# bottom of the image. The lessons indicated that the lanes were about
# 3.7 meters wide with a visible distance of 30 meters. I've used this
# information to determine the appropriate value of delta_y as well.
perspective_delta_x = 744
perspective_delta_y = int(perspective_delta_x * 30 / 3.7)
perspective_border_x = int(perspective_delta_x * 0.34)
perspective_max_y = perspective_delta_y
perspective_max_x = int(perspective_delta_x + 2 * perspective_border_x)
perspective_pixels_per_meter = perspective_delta_x / 3.7

perspective_origin_y_top = 440
perspective_origin_y_bottom = 670
perspective_origin_x_top_left = 609
perspective_origin_x_top_right = 673
perspective_origin_x_bottom_left = 289
perspective_origin_x_bottom_right = 1032

perspective_origin_delta_x_bottom = perspective_origin_x_bottom_right - perspective_origin_x_bottom_left
perspective_origin_delta_x_top = perspective_origin_x_top_right - perspective_origin_x_top_left
perspective_origin_delta_y = perspective_origin_y_bottom - perspective_origin_y_top

def draw_lines_on_dash(dash_img, lines):
  max_y_idx = 5
  points_left = []
  points_right = []
  for line_idx in range(2):
    line = lines[line_idx]
    for y_idx in range(max_y_idx + 1):
      portion_bottom_to_top = y_idx * 1.0 / max_y_idx
      y = perspective_origin_y_bottom - portion_bottom_to_top * perspective_origin_delta_y
      yp = perspective_delta_y * (1 - portion_bottom_to_top)
      xp = line[0] * yp**2 + line[1] * yp + line[2]
      x_portion = (xp - perspective_border_x) / perspective_delta_x
      x_left = perspective_origin_x_bottom_left + portion_bottom_to_top * (perspective_origin_x_top_left - perspective_origin_x_bottom_left)
      x_right = perspective_origin_x_bottom_right + portion_bottom_to_top * (perspective_origin_x_top_right - perspective_origin_x_bottom_right)
      x = x_left + x_portion * (x_right - x_left)
      if line_idx == 0:
        points_left.append([x,y])
      else:
        points_right.append([x,y])
  points_right.reverse()
  points = points_left + points_right
  img = np.zeros_like(dash_img)
  cv2.fillPoly(img, np.int_([points]), (0,255,0))
  res = cv2.addWeighted(dash_img, 1, img, 0.3, 0)
  return res

def perspective_matrices():
  # These points (list of [x,y] pairs) are taken from lane lines
  # in output_images/dash_undistort/straight_lines2.jpg.
  src = np.float32(
          [[perspective_origin_x_top_left,perspective_origin_y_top],       [perspective_origin_x_top_right,perspective_origin_y_top],
           [perspective_origin_x_bottom_left,perspective_origin_y_bottom], [perspective_origin_x_bottom_right,perspective_origin_y_bottom]])
  # These points represent points in the perspective transformed
  # image corresponding to the src points taken from the undistorted image.
  dst = np.float32(
          [[perspective_border_x, 0],
           [perspective_border_x + perspective_delta_x, 0],
           [perspective_border_x, perspective_delta_y],
           [perspective_border_x + perspective_delta_x, perspective_delta_y]])
  M = cv2.getPerspectiveTransform(src,dst)
  M_inv= cv2.getPerspectiveTransform(dst,src)
  return (M,M_inv)

M,M_inv = perspective_matrices()

def perspective_transform(img):
  img = cv2.warpPerspective(img, M, (perspective_max_x, perspective_max_y), flags=cv2.INTER_LINEAR)
  return img

def find_lane_lines(img, prev_left=None, prev_right=None, prev_weight=0.8):
  img_center_x = img.shape[1] / 2.0
  center = np.float32([0.0, 0.0, img_center_x]) # default of straight line down center
  lane_line_width = perspective_delta_x * 0.3
  image_center_width = perspective_delta_x * 0.5
  lane_center_width = perspective_delta_x * 0.4
  lane_max_width = perspective_delta_x * 1.25
  default_left_poly = np.float32([0.0, 0.0, perspective_border_x])
  default_right_poly = np.float32([0.0, 0.0, perspective_max_x - perspective_border_x])
  if prev_left != None and prev_right != None:
    center = (prev_left + prev_right) / 2
  if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  lane_pixels = img.nonzero()
  lane_pixels_y = np.array(lane_pixels[0])
  lane_pixels_x = np.array(lane_pixels[1])

  # Starting guess because we'll look for lane marking pixels near that guess.
  left_poly = prev_left
  right_poly = prev_right
  if left_poly is None:
    left_poly = default_left_poly
  if right_poly is None:
    right_poly = default_right_poly

  # Find pixels within lane_uncertainty of the expected lane line positions
  in_left_lane = (lane_line_width / 2 >
                    abs(lane_pixels_x - (left_poly[0] * lane_pixels_y**2 +
                                         left_poly[1] * lane_pixels_y +
                                         left_poly[2])))
  in_right_lane = (lane_line_width / 2 >
                     abs(lane_pixels_x - (right_poly[0] * lane_pixels_y**2 +
                                          right_poly[1] * lane_pixels_y +
                                          right_poly[2])))
  left_of_lane_center = lane_pixels_x < (-lane_center_width / 2 +
                                         center[0] * lane_pixels_y**2 +
                                         center[1] * lane_pixels_y +
                                         center[2])
  right_of_lane_center = lane_pixels_x > (lane_center_width / 2 +
                                         center[0] * lane_pixels_y**2 +
                                         center[1] * lane_pixels_y +
                                         center[2])
  # We need to find left lane on left and right lane on right.
  # Allow small deviation due to curving lanes crossing the center.
  left_side_of_image = lane_pixels_x < (img_center_x - image_center_width / 2)
  right_side_of_image = lane_pixels_x > (img_center_x + image_center_width / 2)
  near_car = ((lane_pixels_x > (img_center_x - lane_max_width / 2)) &
              (lane_pixels_x < (img_center_x + lane_max_width / 2)))
  # Criteria for which lane each pixel belongs in
  left_lane_indices = (in_left_lane & left_side_of_image & left_of_lane_center & near_car).nonzero()[0]
  right_lane_indices = (in_right_lane & right_side_of_image & right_of_lane_center & near_car).nonzero()[0]
  # In case we don't find enough lane pixels in expected place, snap back to default expectation.
  left_poly = default_left_poly
  right_poly = default_right_poly
  # Prepare variables for line fitting.
  left_lane_y = lane_pixels_y[left_lane_indices]
  right_lane_y = lane_pixels_y[right_lane_indices]
  left_lane_x = lane_pixels_x[left_lane_indices]
  right_lane_x = lane_pixels_x[right_lane_indices]
  # If not enough data, fall back on default value of lines straight ahead.
  if len(left_lane_indices) > 20 and len(right_lane_indices) > 20:
    # Distance between y-values is strong indicator of how well fitting will work.
    left_y_spread = np.amax(left_lane_y) - np.amin(left_lane_y)
    right_y_spread = np.amax(right_lane_y) - np.amin(right_lane_y)
    min_y_spread = min(left_y_spread, right_y_spread)
    # With lane pixels at top and bottom, we can fit a parabola.
    if min_y_spread > 2500 and not (prev_left is None) and not (prev_right is None):
      left_poly = np.polyfit(left_lane_y, left_lane_x, 2)
      right_poly = np.polyfit(right_lane_y, right_lane_x, 2)
    # With just two lane markings, a line is the best we can do.
    elif min_y_spread > 800:
      left_poly[1:3] = np.polyfit(left_lane_y, left_lane_x, 1)
      right_poly[1:3] = np.polyfit(right_lane_y, right_lane_x, 1)
    # With only one lane marking, we'll need to assume that the line is vertical.
    else:
      left_poly[2] = np.mean(left_lane_x)
      right_poly[2] = np.mean(right_lane_x)
  # If previous fits available, we'll apply momentum for smoothness.
  if prev_left != None and prev_right != None:
    left_poly = ((prev_left * prev_weight) + (left_poly * (1 - prev_weight)))
    right_poly = ((prev_right * prev_weight) + (right_poly * (1 - prev_weight)))
    return (left_poly, right_poly)
  else:
    return find_lane_lines(img, prev_left=left_poly, prev_right=right_poly, prev_weight=0.0)

def draw_lane_lines(lines):
  img = np.zeros((perspective_max_y,perspective_max_x), dtype="uint8")
  for line in lines:
    y = 1
    prev_x = int(line[0] * y**2 + line[1] * y + line[2])
    prev_y = y
    for i in range(perspective_max_y):
      y = int(perspective_max_y * i / 20)
      x = int(line[0] * y**2 + line[1] * y + line[2])
      if x > 0 and x < perspective_max_x:
        cv2.line(img, (prev_x,prev_y), (x,y), [255,255,255], 15)
      prev_x = x
      prev_y = y
  return img

def convert_lane_heatmap_to_lane_lines_image(img):
  lines = find_lane_lines(img)
  #lines = find_lane_lines(img, prev_left=lines[0], prev_right=lines[1], prev_weight=0.0)
  return draw_lane_lines(lines)

def radius_of_lane_lines(left_lane, right_lane):
  if left_lane == None or right_lane == None:
    return None
  center = (left_lane + right_lane) / 2
  #print("determining radius for " + str(center))
  if abs(center[0]) < 0.000001:
    return None
  radius_pixels = (1 + (2 * center[0] * perspective_max_y + center[1])**2)**1.5 / (-2 * center[0])
  radius_meters = radius_pixels / perspective_pixels_per_meter
  #print("radius is " + str(radius_pixels) + " pixels or " + str(radius_meters) + " meters.")
  return radius_meters

def offset_from_lane_center(left_lane, right_lane):
  if left_lane == None or right_lane == None:
    return 0.0
  center = (left_lane + right_lane) / 2
  lane_offset = center[0] * perspective_max_y**2 + center[1] * perspective_max_y + center[2]
  car_offset = perspective_max_x / 2.0
  #print("Offset... lane: " + str(lane_offset) + " car: " + str(car_offset))
  return (car_offset - lane_offset) / perspective_pixels_per_meter

def annotate_original_image(img, markings_img=None, lane_lines=(None,None)):
  if markings_img != None:
    markings_pink = np.zeros_like(markings_img)
    markings_gray = cv2.cvtColor(markings_img, cv2.COLOR_RGB2GRAY)
    markings_pink[markings_gray > 100] = np.uint8([255,20,147])
    img = cv2.addWeighted(img, 0.8, markings_pink, 1.0, 0.0)
  if lane_lines[0] != None and lane_lines[1] != None:
    radius = radius_of_lane_lines(lane_lines[0], lane_lines[1])
    offset = offset_from_lane_center(lane_lines[0], lane_lines[1])
    radius_text = "Curvature: Straight"
    if radius and abs(radius) > 100 and abs(radius) < 10000:
      radius_direction = "left"
      if radius > 0:
        radius_direction = "right"
      radius_text = "Curvature radius " + str(100 * int(abs(radius) / 100)) + "m to the " + radius_direction
    offset_text = "Offset: Center"
    if abs(offset) > 0.1:
      offset_direction = "left"
      if offset > 0:
        offset_direction = "right"
      offset_text = "Offset: " + str(int(abs(offset * 10)) / 10.0) + "m to the " + offset_direction
    cv2.putText(img, radius_text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
    cv2.putText(img, offset_text, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
    img = draw_lines_on_dash(img, lane_lines)
  return img

class video_processor(object):
  def __init__(self, model, calibration):
    self.recent_markings = []
    self.model = model
    self.calibration = calibration
    self.prev_left = None
    self.prev_right = None

  def process_image(self,img):
    undistorted = undistort(img, self.calibration)
    markings = image_to_lane_markings(undistorted, self.model)
    self.recent_markings.append(markings)
    if len(self.recent_markings) > 30:
      self.recent_markings = self.recent_markings[-30:]
    combined_markings = np.zeros_like(markings)
    for i in np.random.choice(range(len(self.recent_markings)),size=10):
      combined_markings = cv2.addWeighted(combined_markings, 1.0, self.recent_markings[i], 1.0, 0.0)
    birds_eye_markings = perspective_transform(combined_markings)
    lines = find_lane_lines(birds_eye_markings, prev_left=self.prev_left, prev_right=self.prev_right)
    self.prev_left = lines[0]
    self.prev_right = lines[1]
    return annotate_original_image(undistorted, combined_markings, lines)

def process_video(video_path_in, video_path_out, model, calibration):
  clip_in = VideoFileClip(video_path_in)
  processor = video_processor(model=model, calibration=calibration)
  clip_out = clip_in.fl_image(processor.process_image)
  clip_out.write_videofile(video_path_out, audio=False)

def save_examples_from_video():
  video1 = VideoFileClip('project_video.mp4')
  example_seconds = [0,10,20,30,40,50]
  for s in example_seconds:
    video1.save_frame('test_images/video1_' + str(int(s+0.5)) + '.jpg',
                      s)

def main():
  calibration = calibrate_chessboard()
  #undistort_files(calibration, 'camera_cal/calibration*.jpg', 'output_images/chessboard_undistort')
  #save_examples_from_video()
  #undistort_files(calibration, 'test_images/*.jpg', 'output_images/dash_undistort')
  model = create_model()
  #train_model(model, epochs=1000)
  #model.save_weights('model.h5')
  model.load_weights('model.h5')
  #transform_image_files(crop_scale_white_balance, 'test_images/*.jpg', 'output_images/cropped')
  #transform_image_files(uncrop_scale, 'output_images/cropped/*.jpg', 'output_images/uncropped')
  #transform_image_files((lambda img: image_to_lane_markings(img, model)),
  #                      'test_images/*.jpg', 'output_images/markings')
  #transform_image_files(perspective_transform,
  #                      'output_images/dash_undistort/*.jpg',
  #                      'output_images/birds_eye')
  #undistort_files(calibration,
  #                'output_images/markings/*.jpg',
  #                'output_images/undistort_markings')
  #transform_image_files(perspective_transform,
  #                      'output_images/undistort_markings/*.jpg',
  #                      'output_images/birds_eye_markings')
  transform_image_files(convert_lane_heatmap_to_lane_lines_image,
                        'output_images/birds_eye_markings/*.jpg',
                        'output_images/birds_eye_lines')
  transform_image_files(lambda img: video_processor(model=model,calibration=calibration).process_image(img),
                        'test_images/*.jpg',
                        'output_images/final')
  #process_video('project_video.mp4', 'output_images/videos/project_video.mp4', model, calibration)

if __name__ == '__main__':
  main()
