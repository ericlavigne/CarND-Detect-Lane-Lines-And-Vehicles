import numpy as np
import cv2
from glob import glob

from keras.layers.convolutional import Conv2D
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
  color = cv2.imread(fpath)
  gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
  normalized = np.zeros_like(gray)
  normalized[gray > 0] = 1
  return np.stack([normalized, 1 - normalized], axis=-1)

def preprocess_input_image(img):
  """Normalize to [-0.5,0.5] based on lightest and darkest pixel across all channels"""
  low = np.amin(img)
  high = np.amax(img)
  return (((img - low) * 1.0 / (high - low)) - 0.5).astype(np.float32)

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
                metrics=['mean_squared_error'])

tf_one = tf.constant(1.0)
tf_half = tf.constant(0.5)

def tanh_zero_to_one(x):
  return (tf_one + tf.tanh(x)) * tf_half

def create_model():
  """Create neural network model, defining layer architecture."""
  model = Sequential()
  # Convolution2D(output_depth, convolution height, convolution_width, ...)
  model.add(Conv2D(24, (5, 5), padding='same', input_shape=(720,1280,3)))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(24, (5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(18, (5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(10, (5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(2, (5, 5), padding='same', kernel_regularizer=l2(0.01), activation=tanh_zero_to_one))

  compile_model(model)
  return model

def train_model(model, validation_percentage=None, epochs=100):
  """Train the model. With so few examples, I usually prefer
     to use all examples for training. Setting aside some
     examples for validation is supported but not recommended."""
  data = read_training_data()
  if validation_percentage:
    return model.fit(data['x'], data['y'], epochs=epochs, validation_split = validation_percentage / 100.0)
  else:
    return model.fit(data['x'], data['y'], epochs=epochs)

def save_model(model,path='model'):
  """Save model as .h5 and .json files. Specify path without these extensions."""
  with open(path + '.json', 'w') as arch_file:
    arch_file.write(model.to_json())
  model.save_weights(path + '.h5')

def load_model(path='model'):
  """Load model from .h5 and .json files. Specify path without these extensions."""
  with open(path + '.json', 'r') as arch_file:
    model = model_from_json(arch_file.read())
    compile_model(model)
    model.load_weights(path + '.h5')
    return model

def main():
  #calibration = calibrate_chessboard()
  #undistort_files(calibration, 'camera_cal/calibration*.jpg', 'output_images/chessboard_undistort')
  #undistort_files(calibration, 'test_images/*.jpg', 'output_images/dash_undistort')
  model = create_model()
  train_model(model)
  save_model(model)
  model = load_model()

if __name__ == '__main__':
  main()
