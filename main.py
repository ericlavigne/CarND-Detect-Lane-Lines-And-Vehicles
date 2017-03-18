import numpy as np
import cv2
import glob

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
  return calibration

def transform_image_files(transformation, src_pattern, dst_dir):
  src_fpaths = glob.glob(src_pattern)
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

def main():
  calibration = calibrate_chessboard()
  undistort_files(calibration, 'camera_cal/calibration*.jpg', 'output_images/chessboard_undistort')
  undistort_files(calibration, 'test_images/*.jpg', 'output_images/dash_undistort')

if __name__ == '__main__':
  main()
