import cv2
from glob import glob
import numpy as np

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.regularizers import l2

from moviepy.editor import VideoFileClip

import tensorflow as tf

def read_image(path):
  """Ensure images read in RGB format for consistency with moviepy"""
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def write_image(path,img):
  """Handles RGB or grayscale images"""
  if len(img.shape) == 3 and img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  cv2.imwrite(path, img)

def calibrate_chessboard():
  """Perform calibration using chessboard images"""
  objp = np.zeros((6*9,3), np.float32)
  objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

  calibration_fnames = glob('camera_cal/calibration*.jpg')
  
  calibration_images = []
  objpoints = []
  imgpoints = []
  
  for fname in calibration_fnames:
    img = read_image(fname)
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
  """Concise testing for image transformation functions"""
  src_fpaths = glob(src_pattern)
  for src_fpath in src_fpaths:
    img = read_image(src_fpath)
    dst_img = transformation(img)
    fname = src_fpath.split('/')[-1]
    dst_fpath = dst_dir + '/' + fname
    write_image(dst_fpath,dst_img)

def undistort(img, calibration):
  """Use calibration to correct image distortions"""
  return cv2.undistort(img, calibration[0], calibration[1], None, calibration[0])

def undistort_files(calibration, src_pattern, dst_dir):
  """Test image distortion correction on test files"""
  transform_image_files((lambda x: undistort(x, calibration)), src_pattern, dst_dir)

original_max_x = 1280
original_max_y = 720

lane_settings = {'name': 'lanes',
                 'presence_weight': 50.0, 'threshold': 0.5,
                 'original_max_x': 1280, 'original_max_y': 720,
                 'crop_min_x': 200, 'crop_max_x': 1080,
                 'crop_min_y': 420, 'crop_max_y': 666,
                 'scale_factor': 2}

car_settings = {'name': 'cars',
                'presence_weight': 50.0, 'threshold': 0.5,
                'original_max_x': 1280, 'original_max_y': 720,
                'crop_min_x': 0, 'crop_max_x': 1280,
                'crop_min_y': 420, 'crop_max_y': 666,
                'scale_factor': 2}

def read_training_data_paths():
  """Returns {'x': [path1, path2, ...], 'lanes': [path1, path2, ...], 'cars': [path1, path2, ...]}"""
  x = glob('training/*_x.png')
  lanes = glob('training/*_lanes.png')
  cars = glob('training/*_cars.png')
  x.sort()
  lanes.sort()
  cars.sort()
  assert (len(x) == len(lanes)), "x and lanes files don't match"
  assert (len(x) == len(cars)), "x and cars files don't match"
  return {'x': x, 'lanes': lanes, 'cars': cars}

def read_training_file(fpath,opt):
  """Read (car or lane) annotation file and convert to y format: one channel with
     1 for present or 0 for absent"""
  img = read_image(fpath)
  img = crop_scale_white_balance(img,opt)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  normalized = np.zeros_like(img)
  normalized[img > 0] = 1
  return np.stack([normalized], axis=-1)

def crop(img,opt):
  return img[opt['crop_min_y']:opt['crop_max_y'], opt['crop_min_x']:opt['crop_max_x']]

def uncrop(img,opt):
  target_shape = (opt['original_max_y'],opt['original_max_x'], 3)
  frame = np.zeros(target_shape, dtype="uint8")
  frame[opt['crop_min_y']:opt['crop_max_y'], opt['crop_min_x']:opt['crop_max_x'], 0:3] = img
  img = frame
  return img

def scale_white_balance(img,opt):
  img = cv2.resize(img, None, fx=(1.0/opt['scale_factor']), fy=(1.0/opt['scale_factor']),
                   interpolation=cv2.INTER_AREA)
  low = np.amin(img)
  high = np.amax(img)
  img = (((img - low + 1.0) * 252.0 / (high - low)) - 0.5).astype(np.uint8)
  return img

def unscale(img,opt):
  img = cv2.resize(img, None, fx=opt['scale_factor'], fy=opt['scale_factor'])
  if len(img.shape) == 2:
    img = cv2.merge((img,img,img))
  return img

def crop_scale_white_balance(img,opt):
  img = crop(img,opt)
  img = scale_white_balance(img,opt)
  return img

def uncrop_scale(img,opt):
  img = unscale(img,opt)
  img = uncrop(img,opt)
  return img

def preprocess_input_image(img,opt):
  img = crop_scale_white_balance(img,opt)
  img = cv2.GaussianBlur(img, (3,3), 0)
  return ((img / 253.0) - 0.5).astype(np.float32)

def read_training_data(opt):
  """Returns tuple of input matrix and output matrix (X,y)"""
  paths = read_training_data_paths()
  X = []
  for x in paths['x']:
    X.append(preprocess_input_image(read_image(x), opt))
  Y = []
  for y in paths[opt['name']]:
    Y.append(read_training_file(y,opt))
  return {'x': np.stack(X), 'y': np.stack(Y)}

def weighted_binary_crossentropy(weight):
  """Higher weights increase the importance of examples in which
     the correct answer is 1. Higher values should be used when
     1 is a rare answer. Lower values should be used when 0 is
     a rare answer."""
  return (lambda y_true, y_pred: tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight))

def compile_model(model,opt):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer='adam',
                loss=weighted_binary_crossentropy(opt['presence_weight']),
                metrics=['binary_accuracy', 'binary_crossentropy'])

tf_pos_tanh_offset = tf.constant(0.5)
tf_pos_tanh_scale = tf.constant(0.45)

def tanh_zero_to_one(x):
  """Actually [0.05, 0.95] to avoid divide by zero errors"""
  return (tf.tanh(x) * tf_pos_tanh_scale) + tf_pos_tanh_offset

def create_model(opt):
  """Create neural network model, defining layer architecture."""
  model = Sequential()
  # Convolution2D(output_depth, convolution height, convolution_width, ...)
  model.add(Convolution2D(20, 5, 5, border_mode='same',
            input_shape=(int((opt['crop_max_y'] - opt['crop_min_y']) / opt['scale_factor']),
                         int((opt['crop_max_x'] - opt['crop_min_x']) / opt['scale_factor']),
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
  compile_model(model, opt)
  return model

def train_model(model, opt, validation_percentage=None, epochs=100):
  """Train the model. With so few examples, I usually prefer
     to use all examples for training. Setting aside some
     examples for validation is supported but not recommended."""
  data = read_training_data(opt)
  if validation_percentage:
    return model.fit(data['x'], data['y'], nb_epoch=epochs, validation_split = validation_percentage / 100.0)
  else:
    return model.fit(data['x'], data['y'], nb_epoch=epochs)

def image_to_prediction(img, model, opt):
  model_input = preprocess_input_image(img,opt)[None, :, :, :]
  model_output = model.predict(model_input, batch_size=1)[0]
  odds = cv2.split(model_output)[0]

  threshold = opt['threshold']
  result = np.zeros_like(odds)
  result[odds > threshold] = 254
  
  result = uncrop_scale(result,opt)
  return result

# These parameters control both the size and vertical scaling of the image.
# I chose delta_x based on the width in pixels of the lane lines at the
# bottom of the image. The lessons indicated that the lanes were about
# 3.7 meters wide with a visible distance of 30 meters. I've used this
# information to determine the appropriate value of delta_y as well.
perspective_delta_x = 744
perspective_delta_y = int(perspective_delta_x * 30 / 3.7)
perspective_border_x = int(perspective_delta_x * 0.7)
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

def perspective_reverse(img):
  img = cv2.warpPerspective(img, M_inv, (original_max_x, original_max_y), flags=cv2.INTER_LINEAR)
  return img

def find_lane_centroids(img):
  # Assuming usual lane width and car in center of lane, these are likely places to find bottom of lane lines
  expected_x_starts = [perspective_border_x, perspective_border_x + perspective_delta_x]
  # Create two lists to contain centroids for left and right lane lines
  centroids = [[],[]]
  # Size of squares in which we'll search. Wide enough to handle uncertainty.
  # Narrow enough not to pick up the wrong lane line.
  search_range = int(perspective_delta_x / 3.0)
  y_iterations = int(perspective_max_y * 0.5 / search_range)
  # Which pixels in the image have been identified as likely lane markings?
  lane_pixels = img.nonzero()
  lane_pixels_y = np.array(lane_pixels[0])
  lane_pixels_x = np.array(lane_pixels[1])
  # For each lane, sweep from bottom of image to top. We're already fairly
  # certain where lanes start at bottom.
  for lane_idx in range(2):
    last_x = expected_x_starts[lane_idx]
    found_first = False
    for y_idx in range(y_iterations):
      y_mid = int((y_iterations - y_idx) * perspective_max_y / y_iterations)
      y_min = y_mid - search_range
      y_max = y_mid + search_range
      x_min = last_x - search_range
      x_max = last_x + search_range
      found_indices = ((lane_pixels_x >= x_min) & (lane_pixels_x <= x_max) & (lane_pixels_y >= y_min) & (lane_pixels_y <= y_max)).nonzero()[0]
      found_x = lane_pixels_x[found_indices]
      if len(found_x) > 1:
        last_x = int(np.mean(found_x))
        found_first = True
      if found_first:
        centroids[lane_idx].append([last_x, y_mid])
  return centroids

def draw_lane_centroids(img, centroids):
  img = np.copy(img)
  for lane_idx in range(2):
    for center in centroids[lane_idx]:
      cv2.circle(img, (center[0],center[1]), 20, (255,255,255), 10)
  return img

def fit_parabolas_to_lane_centroids(centroids):
  polys = []
  for lane_idx in range(2):
    x_vals = []
    y_vals = []
    for point in centroids[lane_idx]:
      x_vals.append(point[0])
      y_vals.append(point[1])
    min_y = np.amin(y_vals)
    max_y = np.amax(y_vals)
    mid_y = (min_y + max_y) / 2
    weights = []
    for y in y_vals:
      if y > mid_y:
        weights.append(1.0)
      else:
        weights.append(max(0.1, ((y - min_y) * 1.0 / (mid_y - min_y))))
    polys.append(np.polyfit(y_vals, x_vals, 2, w=weights))
  return polys

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

def draw_lane_lines(lines):
  img = np.zeros((perspective_max_y, perspective_max_x, 3), dtype="uint8")
  points = [[],[]]
  for line_idx in range(2):
    line = lines[line_idx]
    for i in range(31):
      y = int(perspective_max_y * i / 30)
      x = int(line[0] * y**2 + line[1] * y + line[2])
      points[line_idx].append((x,y))
  points[1].reverse()
  cv2.fillPoly(img, np.int_([points[0] + points[1]]), (0,255,0))
  return img

def draw_lines_on_dash(dash_img, lines):
  perspective_lanes_img = draw_lane_lines(lines)
  dash_lanes_img = perspective_reverse(perspective_lanes_img)
  res = cv2.addWeighted(dash_img, 1, dash_lanes_img, 0.3, 0)
  return res

def convert_lane_heatmap_to_lane_lines_image(img):
  centroids = find_lane_centroids(img)
  lines = fit_parabolas_to_lane_centroids(centroids)
  res = draw_lane_lines(lines)
  res = draw_lane_centroids(res, centroids)
  return res

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

def annotate_original_image(img, lane_markings_img=None, lane_lines=(None,None), car_img=None):
  if lane_markings_img != None:
    markings_pink = np.zeros_like(lane_markings_img)
    markings_gray = cv2.cvtColor(lane_markings_img, cv2.COLOR_RGB2GRAY)
    markings_pink[markings_gray > 100] = np.uint8([255,20,147])
    img = cv2.addWeighted(img, 0.8, markings_pink, 1.0, 0.0)
  if car_img != None:
    car_cyan = np.zeros_like(car_img)
    car_gray = cv2.cvtColor(car_img, cv2.COLOR_RGB2GRAY)
    car_cyan[car_gray > 100] = np.uint8([0,255,255])
    img = cv2.addWeighted(img, 0.8, car_cyan, 0.5, 0.0)
  if lane_lines[0] != None and lane_lines[1] != None:
    radius = radius_of_lane_lines(lane_lines[0], lane_lines[1])
    offset = offset_from_lane_center(lane_lines[0], lane_lines[1])
    radius_text = "Curvature: Straight"
    if radius and abs(radius) > 100 and abs(radius) < 10000:
      radius_direction = "right"
      if radius > 0:
        radius_direction = "left"
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
  def __init__(self, lane_model, car_model, calibration):
    self.recent_markings = []
    self.lane_model = lane_model
    self.car_model = car_model
    self.calibration = calibration
    self.prev_left = None
    self.prev_right = None

  def process_image(self,img):
    undistorted = undistort(img, self.calibration)
    markings = image_to_prediction(undistorted, self.lane_model, lane_settings)
    cars = image_to_prediction(undistorted, self.car_model, car_settings)
    self.recent_markings.append(markings)
    if len(self.recent_markings) > 30:
      self.recent_markings = self.recent_markings[-30:]
    combined_markings = np.zeros_like(markings)
    included_so_far = 0
    for i in np.random.choice(range(len(self.recent_markings)),size=10):
      new_weight = 1.0 / (included_so_far + 1)
      old_weight = 1.0 - new_weight
      combined_markings = cv2.addWeighted(combined_markings, old_weight , self.recent_markings[i], new_weight, 0.0)
      included_so_far += 1
    combined_markings[combined_markings < 80] = 0
    birds_eye_markings = perspective_transform(combined_markings)
    #print("=== Processing image ===")
    centroids = find_lane_centroids(birds_eye_markings)
    #print("Centroids: " + str(centroids))
    lines = fit_parabolas_to_lane_centroids(centroids)
    #print("Lines: " + str(lines))
    self.prev_left = lines[0]
    self.prev_right = lines[1]
    result = annotate_original_image(undistorted, combined_markings, lines, cars)
    #print("++++++++++++++++++++++++")
    return result

def process_video(video_path_in, video_path_out, lane_model, car_model, calibration):
  clip_in = VideoFileClip(video_path_in)
  processor = video_processor(lane_model=lane_model, car_model=car_model, calibration=calibration)
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
  
  lane_model = create_model(lane_settings)
  #train_model(lane_model, lane_settings, epochs=1000)
  #lane_model.save_weights('models/lanes.h5')
  lane_model.load_weights('models/lanes.h5')
  
  car_model = create_model(car_settings)
  #train_model(car_model, car_settings, epochs=1000)
  #car_model.save_weights('models/cars.h5')
  car_model.load_weights('models/cars.h5')
  
  #transform_image_files(lambda img: crop_scale_white_balance(img, lane_settings),
  #                      'test_images/*.jpg', 'output_images/cropped_lanes')
  #transform_image_files(lambda img: uncrop_scale(img, lane_settings),
  #                      'output_images/cropped_lanes/*.jpg', 'output_images/uncropped_lanes')
  #transform_image_files((lambda img: image_to_prediction(img, lane_model, lane_settings)),
  #                      'test_images/*.jpg', 'output_images/markings')
  transform_image_files(perspective_transform,
                        'output_images/dash_undistort/*.jpg',
                        'output_images/birds_eye')
  transform_image_files(perspective_reverse,
                        'output_images/birds_eye/*.jpg',
                        'output_images/bird_to_dash')
  undistort_files(calibration,
                  'output_images/markings/*.jpg',
                  'output_images/undistort_markings')
  transform_image_files(perspective_transform,
                        'output_images/undistort_markings/*.jpg',
                        'output_images/birds_eye_markings')
  transform_image_files(convert_lane_heatmap_to_lane_lines_image,
                        'output_images/birds_eye_markings/*.jpg',
                        'output_images/birds_eye_lines')
  transform_image_files(lambda img: video_processor(lane_model=lane_model,car_model=car_model,calibration=calibration).process_image(img),
                        'test_images/*.jpg',
                        'output_images/final')
  #process_video('project_video.mp4', 'output_images/videos/project_video.mp4', lane_model, car_model, calibration)

if __name__ == '__main__':
  main()
