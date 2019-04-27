import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

'''The code has been referred from:
 https://becominghuman.ai/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-i-c013907d1ab5'''
# make sure that the pictures taken have good illumination, and that the pattern is captured from different angles.
# Also make sure that the pattern is located in different parts of the screen.
# Bad calibration can happen if you only take pictures that are centered. Make sure there’s a lot of variation in the pictures.

#It is important to point out that not all pictures will be suitable for detecting the pattern. It is hard to know
# beforehand which pictures will work and therefore it is a good idea to take as many pictures as possible. I took 34.

#The first step is to pick the chessboard size. While the size is completely arbitrary, it is recommended to pick
# an asymmetrical size (ie. a rectangle, not a square). I chose 9x7


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#The second step is to define a grid where all the points will be stored. This is to store the points in an ordered
# manner, like so: (0,0,0), (1,0,0), (2,0,0) ….,(6,5,0)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob(r'C:\Users\vedan\Desktop\Computer Vision\Assignment 3\Camera Calibration\pics\Originals\GO*.JPG')

#After loading the image we have to convert it to grayscale and then we use the findChessboardCorners algorithm.
#This algorithm will return the corners detected and a flag called ret that will be true if the algorithm
#was able to detect the pattern.

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
# FINDCHESSBOARDCORNERS  Finds the positions of internal corners of the chessboard
    #
    #      [corners,ok] = cv.findChessboardCorners(im, patternSize)
    #      [...] = cv.findChessboardCorners(..., 'OptionName', optionValue, ...)
    #
    #  ## Input
    #  * __im__ Source chessboard view. It must be an 8-bit grayscale or color
    #    image.
    #  * __patternSize__ Number of inner corners per a chessboard row and column
    #    (`patternSize = [points_per_row,points_per_colum] = [columns,rows]`).
    #
    #  ## Output
    #  * __corners__ Output array of detected corners. Cell array of 2-element
    #    vectors `{[x,y], ...}`. Returns an empty cell if it fails to find all the
    #    corners.
    #  * __ok__ returns true if all of the corners are found and they are placed in
    #    a certain order. Otherwise, if the function fails to find all the corners
    #    or reorder them, it returns false.
    #
    #  ## Options
    #  * __AdaptiveThresh__ Use adaptive thresholding to convert the image to black
    #    and white, rather than a fixed threshold level (computed from the average
    #    image brightness). default true.
    #  * __NormalizeImage__ Normalize the image gamma with cv.equalizeHist before
    #    applying fixed or adaptive thresholding. default true.
    #  * __FilterQuads__ Use additional criteria (like contour area, perimeter,
    #    square-like shape) to filter out false quads extracted at the contour
    #    retrieval stage. default false.
    #  * __FastCheck__ Run a fast check on the image that looks for chessboard
    #    corners, and shortcut the call if none is found. This can drastically
    #    speed up the call in the degenerate condition when no chessboard is
    #    observed. default false.
    #
    #  The function attempts to determine whether the input image is a view of the
    #  chessboard pattern and locate the internal chessboard corners. The function
    #  returns a non-zero value if all of the corners are found and they are placed
    #  in a certain order (row by row, left to right in every row). Otherwise, if
    #  the function fails to find all the corners or reorder them, it returns an
    #  empty cell array. For example, a regular chessboard has 8x8 squares and 7x7
    #  internal corners, that is, points where the black squares touch each other.
    #  The detected coordinates are approximate, and to determine their positions
    #  more accurately, the function calls cv.cornerSubPix. You also may use the
    #  function cv.cornerSubPix with different parameters if returned coordinates
    #  are not accurate enough.
    #
    #  In practice, it is more convenient to use a chessboard grid that is
    #  asymmetric, for example 5x6. Using such even-odd asymmetry yields a
    #  chessboard that has only one symmetry axis, so the board orientation can
    #  always be defined uniquely.

    ret, corners = cv2.findChessboardCorners(gray, (9,7), None)

# In order to improve the accuracy in calibration algorithm, refine the location of the corners to subpixel accuracy.
# In this case we have to define the criteria required to locate.

#Criteria is defined as follows criteria = (type, number of iterations, accuracy). In this case we are telling the
# algorithm that we care both about number of iterations and accuracy (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER)
# and we selected 30 iterations and an accuracy of 0.001, It is defined above in line 18.

    if ret == True:
        corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    # # If found, add object points, image points
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,7), corners, ret)
        # DRAWCHESSBOARDCORNERS  Renders the detected chessboard corners
        #
        #      im = cv.drawChessboardCorners(im, patternSize, corners)
        #      im = cv.drawChessboardCorners(..., 'OptionName', optionValue, ...)
        #
        #  ## Input
        #  * __im__ Source image. It must be an 8-bit color image.
        #  * __patternSize__ Number of inner corners per a chessboard row and column
        #    (`patternSize = [points_per_row, points_per_column]`).
        #  * __corners__ Array of detected corners, the output of
        #    cv.findChessboardCorners.
        #
        #  ## Output
        #  * __im__ Destination image.
        #
        #  ## Options
        #  * __PatternWasFound__ Parameter indicating whether the complete board was
        #    found or not. The return value of cv.findChessboardCorners should be
        #    passed here. default true
        #
        #  The function draws individual chessboard corners detected either as red
        #  circles if the board was not found, or as colored corners connected with
        #  lines if the board was found.

        write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        #cv2.imshow('img', img)
        #cv2.waitKey(5)

cv2.destroyAllWindows()

# Test undistortion on an image
img = cv2.imread('GOPR001.JPG')
plt.imshow(img), plt.show
h,  w = img.shape[:2]

# After analyzing all the pictures, we then run the cv2.calibrateCamera algorithm. This is the algorithm that outputs
# the camera parameters. This algorithm returns the camera matrix (K) distortion coefficients (dist) and the rotation
# and translation vectors (rvecs and tvecs).
# Do camera calibration given object points and image points and save the results. A detailed explanation of the calibration parameters
# is given in the report in the Camera Calibration section.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h),None,None)

# CALIBRATECAMERA  Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern
#
#      [cameraMatrix, distCoeffs, reprojErr] = cv.calibrateCamera(objectPoints, imagePoints, imageSize)
#      [cameraMatrix, distCoeffs, reprojErr, rvecs, tvecs, stdDevsIntrinsics, stdDevsExtrinsics, perViewErrors] = cv.calibrateCamera(...)
#      [...] = cv.calibrateCamera(..., 'OptionName', optionValue, ...)
#
#  ## Input
#  * __objectPoints__ A cell array of cells of calibration pattern points in
#    the calibration pattern coordinate space `{{[x,y,z], ..}, ...}`. The outer
#    vector contains as many elements as the number of the pattern views. If
#    the same calibration pattern is shown in each view and it is fully
#    visible, all the vectors will be the same. Although, it is possible to use
#    partially occluded patterns, or even different patterns in different
#    views. Then, the vectors will be different. The points are 3D, but since
#    they are in a pattern coordinate system, then, if the rig is planar, it
#   may make sense to put the model to a XY coordinate plane so that
#    Z-coordinate of each input object point is 0. Requires at least 4 points
#    per view.
#  * __imagePoints__ A cell array of cells of the projections of calibration
#    pattern points `{{[x,y], ..}, ...}`. `numel(imagePoints)` and
#    `numel(objectPoints)` must be equal, and `numel(imagePoints{i})` must be
#    equal to `numel(objectPoints{i})` for each `i`.
#  * __imageSize__ Size of the image used only to initialize the intrinsic
#    camera matrix `[w,h]`.
#
#  ## Output
#  * __cameraMatrix__ Output 3x3 floating-point camera matrix
#    `A = [fx 0 cx; 0 fy cy; 0 0 1]`
#  * __distCoeffs__ Output vector of distortion coefficients
#    `[k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy]` of 4, 5, 8, 12 or 14
#    elements.
#  * __reprojErr__ the overall RMS re-projection error.
#  * __rvecs__ Output cell array of rotation vectors (see cv.Rodrigues)
#    estimated for each pattern view (cell array of 3-element vectors). That
#    is,  each k-th rotation vector together with the corresponding k-th
#    translation vector (see the next output parameter description) brings the
#    calibration pattern from the model coordinate space (in which object
#    points are specified) to the world coordinate space, that is, a real
#    position of the calibration pattern in the k-th pattern view (`k=1:M`)
#  * __tvecs__ Output cell array of translation vectors estimated for each
#    pattern view (cell array of 3-element vectors).
#  * __stdDevsIntrinsics__ Output vector of standard deviations estimated for
#    intrinsic parameters. Order of deviations values:
#    `(fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy)`. If one of
#    parameters is not estimated, its deviation is equals to zero.
#  * __stdDevsExtrinsics__ Output vector of standard deviations estimated for
#    extrinsic parameters. Order of deviations values: `(R1, T1, ..., RM, TM)`
#    where `M` is number of pattern views, `Ri, Ti` are concatenated 1x3
#    vectors.
#  * __perViewErrors__ Output vector of the RMS re-projection error estimated
#    for each pattern view.

# The function estimates the intrinsic camera parameters and extrinsic
#  parameters for each of the views. The algorithm is based on [Zhang2000] and
#  [BoughuetMCT]. The coordinates of 3D object points and their corresponding
#  2D projections in each view must be specified. That may be achieved by using
#  an object with a known geometry and easily detectable feature points. Such
#  an object is called a calibration rig or calibration pattern, and OpenCV has
#  built-in support for a chessboard as a calibration rig (see
#  cv.findChessboardCorners). Currently, initialization of intrinsic parameters
#  (when `UseIntrinsicGuess` is not set) is only implemented for planar
#  calibration patterns (where Z-coordinates of the object points must be all
#  zeros). 3D calibration rigs can also be used as long as initial
# `CameraMatrix` is provided.
#
#  The algorithm performs the following steps:
#
#  1. Compute the initial intrinsic parameters (the option only available for
#     planar calibration patterns) or read them from the input parameters.
#     The distortion coefficients are all set to zeros initially unless some
#     of 'FixK?' are specified.
#  2. Estimate the initial camera pose as if the intrinsic parameters have
#     been already known. This is done using cv.solvePnP.
#  3. Run the global Levenberg-Marquardt optimization algorithm to minimize
#     the reprojection error, that is, the total sum of squared distances
#     between the observed feature points `imagePoints` and the projected
#     (using the current estimates for camera parameters and the poses)
#     object points `objectPoints`.
np.savez(r'C:\Users\vedan\Desktop\Computer Vision\Assignment 3\Camera Calibration\pics\calib.npz',mtx=mtx,rvecs=rvecs,dist=dist, tvecs=tvecs )
print ('mtx=', mtx)
print('dist=', dist)
print('rvecs=',rvecs)
print('tvecs=', tvecs)

# Undistort image. By obtaining a new camera matrix we test undistortion on a test image just to test whether we have
# good distortion coefficients.
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# GETOPTIMALNEWCAMERAMATRIX  Returns the new camera matrix based on the free scaling parameter
#
#      cameraMatrix = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize)
#      [cameraMatrix, validPixROI] = cv.getOptimalNewCameraMatrix(...)
#      [...] = cv.getOptimalNewCameraMatrix(..., 'OptionName', optionValue, ...)
#
#  ## Input
#  * __cameraMatrix__ Input 3x3 camera matrix, `A = [fx 0 cx; 0 fy cy; 0 0 1]`.
#  * __distCoeffs__ Input vector of distortion coefficients
#    `[k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy]` of 4, 5, 8, 12 or 14
#    elements. If the vector is empty, the zero distortion coefficients are
#    assumed.
#  * __imageSize__ Original image size `[w,h]`.
#
#  ## Output
#  * __cameraMatrix__ Output new camera matrix, 3x3.
#  * __validPixROI__ Optional output rectangle `[x,y,w,h]` that outlines
#    all-good-pixels region in the undistorted image. See `roi1`, `roi2`
#    description in cv.stereoRectify.

# The function computes and returns the optimal new camera matrix based on the
# free scaling parameter. By varying this parameter, you may retrieve only
# sensible pixels `Alpha=0`, keep all the original image pixels if there is
# valuable information in the corners `Alpha=1`, or get something in between.
# When `Alpha>0`, the undistorted result is likely to have some black pixels
# corresponding to "virtual" pixels outside of the captured distorted image.
# The original camera matrix, distortion coefficients, the computed new camera
# matrix, and `newImageSize` should be passed to cv.initUndistortRectifyMap to
# produce the maps for cv.remap.
np.savez('newcameramtx.npz',newcameramtx) # saves the new camera matrix
print(newcameramtx)
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
cv2.imwrite(r'C:\Users\vedan\Desktop\Computer Vision\Assignment 3\Camera Calibration\Undistorted.jpg',dst)

