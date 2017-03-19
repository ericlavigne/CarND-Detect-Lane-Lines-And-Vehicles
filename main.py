import numpy as np
import cv2
from glob import glob

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.regularizers import l2

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
  return np.stack([normalized, 1 - normalized], axis=-1)

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
                        #(int(img.shape[1] / scale_factor),
                        # int(img.shape[0] / scale_factor)),
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
  #if len(img.shape) == 2:
  #  target_shape = (original_max_y, original_max_x, img.shape[2])
  frame = np.zeros(target_shape, dtype="uint8")
  frame[crop_min_y:crop_max_y, crop_min_x:crop_max_x, 0:3] = img
  img = frame
  return img

def preprocess_input_image(img):
  """Normalize to [-0.5,0.5] based on lightest and darkest pixel across all channels"""
  img = crop_scale_white_balance(img)
  img = cv2.GaussianBlur(img, (3,3), 0)
  b,g,r = cv2.split(img)
  x = np.zeros_like(b)
  y = np.zeros_like(b)
  imax,jmax = b.shape
  for i in range(imax):
    for j in range(jmax):
      x[i][j] = int(i * 253.0 / imax + 0.5)
      y[i][j] = int(j * 253.0 / jmax + 0.5)
  img = cv2.merge((b,g,r,x,y))
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

def compile_model(model):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])

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
                         5)))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Convolution2D(20, 5, 5, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Convolution2D(20, 5, 5, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Convolution2D(20, 5, 5, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Convolution2D(2, 5, 5, border_mode='same', W_regularizer=l2(0.01), activation=tanh_zero_to_one))

  compile_model(model)
  return model

def train_model(model, validation_percentage=None, epochs=100):
  """Train the model. With so few examples, I usually prefer
     to use all examples for training. Setting aside some
     examples for validation is supported but not recommended."""
  data = read_training_data()
  if validation_percentage:
    return model.fit(data['x'], data['y'], nb_epoch=epochs, batch_size=1, validation_split = validation_percentage / 100.0)
  else:
    return model.fit(data['x'], data['y'], nb_epoch=epochs, batch_size=1)

def image_to_lane_lines_mask(img, model, threshold=0.5):
  model_input = preprocess_input_image(img)[None, :, :, :]
  model_output = model.predict(model_input, batch_size=1)[0]
  lane_line_odds, not_lane_line_odds = cv2.split(model_output)
  result = np.zeros_like(lane_line_odds)
  result[lane_line_odds > threshold] = 254
  result = uncrop_scale(result)
  return result

def main():
  #calibration = calibrate_chessboard()
  #undistort_files(calibration, 'camera_cal/calibration*.jpg', 'output_images/chessboard_undistort')
  #undistort_files(calibration, 'test_images/*.jpg', 'output_images/dash_undistort')
  model = create_model()
  train_model(model, epochs=10)
  model.save_weights('model.h5')
  model.load_weights('model.h5')
  transform_image_files(crop_scale_white_balance, 'test_images/*.jpg', 'output_images/cropped')
  transform_image_files(uncrop_scale, 'output_images/cropped/*.jpg', 'output_images/uncropped')
  transform_image_files((lambda img: image_to_lane_lines_mask(img, model, threshold=0.5)),
                        'test_images/*.jpg', 'output_images/lane_lines')

if __name__ == '__main__':
  main()
