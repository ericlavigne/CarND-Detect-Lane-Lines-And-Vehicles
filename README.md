## Detecting Lane Lines and Vehicles
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project satisfies the requirements for both the Advanced Lane Finding project
and the Vehicle Detection project for Udacity's Self-Driving Car Engineer
nanodegree. Primary goals include detecting the lane lines, determining the
curvature of the lane as well as the car's position within the lane, and
detecting other vehicles.

Find the latest version of this project on
[Github](https://github.com/ericlavigne/CarND-Detect-Lane-Lines-And-Vehicles).

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Detect vehicle pixels and place bounding boxes around each detected vehicle.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

---

Calibrating the Camera
---

The first step was to calibrate the camera, correcting for distortions. I used
pictures of chessboard taken from various angles, with the assumptions that
the chessboard was made up of perfectly aligned grid squares, to characterize
the lens distortions. I then applied the resulting calibration to the calibration
images to confirm that all visible distortions had been removed. See the
calibrate_chessboard function in
[main.py](https://github.com/ericlavigne/CarND-Detect-Lane-Lines-And-Vehicles/blob/master/main.py).

| Original Image          | Undistorted Image                      |
|:-----------------------:|:--------------------------------------:|
| ![original image](https://raw.githubusercontent.com/ericlavigne/CarND-Detect-Lane-Lines-And-Vehicles/master/camera_cal/calibration1.jpg)          | ![undistorted image](https://raw.githubusercontent.com/ericlavigne/CarND-Detect-Lane-Lines-And-Vehicles/master/output_images/chessboard_undistort/calibration1.jpg)                         |

I applied the same calibration to undistort images from the dashboard camera.
The effect is subtle - note the difference in shape around the left and right
edges of the car hood in the images below.

| Original Image          | Undistorted Image                      |
|:-----------------------:|:--------------------------------------:|
| ![original image](https://raw.githubusercontent.com/ericlavigne/CarND-Detect-Lane-Lines-And-Vehicles/master/test_images/straight_lines1.jpg)          | ![undistorted image](https://raw.githubusercontent.com/ericlavigne/CarND-Detect-Lane-Lines-And-Vehicles/master/output_images/dash_undistort/straight_lines1.jpg)                         |


Installation
---

1. Clone the repository

```sh
git clone https://github.com/ericlavigne/CarND-Detect-Lane-Lines-And-Vehicles
```

2. Setup virtualenv.

```sh
cd CarND-Detect-Lane-Lines-And-Vehicles
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements-cpu.txt (or requirements-gpu.txt if CUDA is available)
deactivate
```

Running the project
---

```sh
cd CarND-Detect-Lane-Lines-And-Vehicles
source env/bin/activate
python main.py
deactivate
```

Installing new library
---

```sh
cd CarND-Detect-Lane-Lines-And-Vehicles
source env/bin/activate
pip freeze > requirements.txt (or requirements-gpu.txt if CUDA is available)
deactivate
```
