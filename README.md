# Camera-Calibration
Download and print out a calibration "chessboard". Take 30 pictures of the board being sure to cover the entire visual eld of the camera. (Not every image need cover the field, but over the thirty, it should be covered). The following page on stackoverflow.com has
an excellent description of a good way to do this:
https://stackoverflow.com/questions/12794876/... ...how-to-verify-the-correctness-of-calibration-of-a-
webcam/12821056#12821056

In this repository:
The python code for calibrating the camera is named as Camera Calibration.py. The code uses at least 20 input images of a chessboard taken from different angles. These images are named as GOPR001.JPG - GOPR020.JPG.

The result of camera calibration is the properties of your camera in the form of matrices. These matrices are named as: camera matrix, distrortion
coefficient matrix, rotational matrix, and translation vectors.

By using opencv's undistort method and the distortion coefficients obtained, we generate a new and more accurate camera matrix. 
All the camera parameters obtained are saved as .npz file.

These parameters act as the starting point for epipolar stereopsis. The epipolar stereopsis can be found in the repository named "Epipolar-Stereopsis".

If you find any problems, please feel free to contact me at vedansh.thakkar@vanderbilt.edu
