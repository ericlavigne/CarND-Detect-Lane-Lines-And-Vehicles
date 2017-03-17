import numpy as np
import cv2
import glob

def undistort(img, calibration):
  return cv2.undistort(img, calibration[0], calibration[1], None, calibration[0])

def calibrate_chessboard():
  objp = np.zeros((6*9,3), np.float32)
  objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

  calibration_fnames = glob.glob('camera_cal/calibration*.jpg')
  
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
  for fname in calibration_fnames:
    img = cv2.imread(fname)
    dst = undistort(img, calibration)
    dst_fname = fname.replace("camera_cal/calibration","output_images/chessboard_undistort/")
    cv2.imwrite(dst_fname, dst)
    
  return calibration

calibration = calibrate_chessboard()

test_fnames = glob.glob('test_images/*.jpg')

for fname in test_fnames:
  img = cv2.imread(fname)
  dst = undistort(img, calibration)
  dst_fname = fname.replace("test_images", "output_images/dash_undistort")
  cv2.imwrite(dst_fname, dst)
