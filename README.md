## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

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

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!


---

Calibrating the Camera
---

The first step was to calibrate the camera, correcting for distortions. I used
pictures of chessboard taken from various angles, with the assumptions that
the chessboard was made up of perfectly aligned grid squares, to characterize
the lens distortions. I then applied the resulting calibration to the calibration
images to confirm that all visible distortions had been removed. See the
calibrate_chessboard function in
[main.py](https://github.com/ericlavigne/CarND-Advanced-Lane-Lines/blob/master/main.py).

| Original Image          | Undistorted Image                      |
|:-----------------------:|:--------------------------------------:|
| ![original image](https://raw.githubusercontent.com/ericlavigne/CarND-Advanced-Lane-Lines/master/camera_cal/calibration1.jpg)          | ![undistorted image](https://raw.githubusercontent.com/ericlavigne/CarND-Advanced-Lane-Lines/master/output_images/chessboard_undistort/1.jpg)                         |

I applied the same calibration to undistort images from the dashboard camera.
The effect is subtle - note the difference in shape around the left and right
edges of the car hood in the images below.

| Original Image          | Undistorted Image                      |
|:-----------------------:|:--------------------------------------:|
| ![original image](https://raw.githubusercontent.com/ericlavigne/CarND-Advanced-Lane-Lines/master/test_images/straight_lines1.jpg)          | ![undistorted image](https://raw.githubusercontent.com/ericlavigne/CarND-Advanced-Lane-Lines/master/output_images/dash_undistort/straight_lines1.jpg)                         |


Installation
---

1. Clone the repository

```sh
git clone https://github.com/ericlavigne/CarND-Advanced-Lane-Lines
```

2. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset of 32x32 traffic sign images, split into training, validation and test sets. Unzip dataset into project directory.

3. Setup virtualenv.

```sh
cd CarND-Advanced-Lane-Lines
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
deactivate
```

Running the project
---

```sh
cd CarND-Advanced-Lane-Lines
source env/bin/activate
python main.py
deactivate
```

Installing new library
---

```sh
cd CarND-Advanced-Lane-Lines
source env/bin/activate
pip freeze > requirements.txt
deactivate
```
