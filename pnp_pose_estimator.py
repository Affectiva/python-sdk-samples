import numpy as np
from numpy import genfromtxt
import os
import cv2

OPENCV_FISHEYE = "fisheye"
OPENCV_PINHOLE = "pinhole"

this_dir = os.path.dirname(__file__)
ourModel_path = this_dir + "/3d_model/model3d_58pts.csv"
OUR_3D_MODEL = genfromtxt(ourModel_path, delimiter=',')
right_eye_left_eye_nose_chin_name_to_index = {"right_eye": 16, "left_eye": 19, "nose": 12, "chin": 2}


def get_default_camera_matrix(calibration_image_resolution=None, current_image_resolution=None):
    """
    This function returns a dummy camera matrix that can be used for quick prototyping
   """
    if calibration_image_resolution is None:
        calibration_image_resolution = np.array([1080, 1920])

    fx = calibration_image_resolution[1]
    fy = fx
    cx = calibration_image_resolution[1]/2.0
    cy = calibration_image_resolution[0]/2.0
    skew = 0
    camera_matrix = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]], dtype="double")

    if current_image_resolution is None:
        current_image_resolution = calibration_image_resolution

    camera_matrix = adjust_camera_matrix(camera_matrix=camera_matrix,
                                         calibration_image_resolution=calibration_image_resolution,
                                         current_image_resolution=current_image_resolution)
    return camera_matrix.copy()


def adjust_camera_matrix(camera_matrix, calibration_image_resolution, current_image_resolution):
    camera_matrix = camera_matrix.copy()
    # check : https://docs.opencv.org/trunk/d9/d0c/group__calib3d.html#ga69f2545a8b62a6b0fc2ee060dc30559d
    camera_matrix[0][0] *= float(current_image_resolution[1]) / calibration_image_resolution[1]
    camera_matrix[0][2] *= float(current_image_resolution[1]) / calibration_image_resolution[1]
    camera_matrix[1][1] *= float(current_image_resolution[0]) / calibration_image_resolution[0]
    camera_matrix[1][2] *= float(current_image_resolution[0]) / calibration_image_resolution[0]
    return camera_matrix


def get_default_distortion_coefficients(camera_type):
    assert camera_type in [OPENCV_FISHEYE, OPENCV_PINHOLE], "Currently, we only support " \
                                                            "opencv_fisheye and opencv_pinhole for the camera_type parameter."

    if camera_type == OPENCV_PINHOLE:
        dist_coeffs = np.zeros((5, 1))
    elif camera_type == OPENCV_FISHEYE:
        dist_coeffs = np.zeros((4, 1))

    return dist_coeffs


def uncalibrated_estimate_pose(n_2d_image_space_points, camera_type=None):
    """
    This function works as a dummy wrapper for the estimate_pose function for quick prototyping.
    It assumes default camera parameters and image resolution.
    It will NOT return accurate results.
   """

    if camera_type is None:
        camera_type = OPENCV_PINHOLE

    current_image_resolution = calibration_image_resolution = np.array([1080, 1920])

    camera_matrix = get_default_camera_matrix(calibration_image_resolution=calibration_image_resolution,
                                              current_image_resolution=current_image_resolution)
    dist_coeffs = get_default_distortion_coefficients(camera_type)

    return estimate_pose_wrapper(n_2d_image_space_points=n_2d_image_space_points, camera_matrix=camera_matrix,
                                 dist_coeffs=dist_coeffs,
                                 current_image_resolution=current_image_resolution,
                                 calibration_image_resolution=calibration_image_resolution,
                                 camera_type=camera_type, n_3d_local_space_points=None)


def estimate_pose_wrapper(n_2d_image_space_points, camera_matrix, current_image_resolution, calibration_image_resolution,
                          camera_type, n_3d_local_space_points=None, dist_coeffs=None):
    """
    This function works as a wrapper for the estimate_pose function. It does  extra things :
        1) It adjusts the intrinsic camera matrix
        2) It is able to load the n_3d_local_space_points from the csv if they are not provided.
    Note that if n_3d_local_space_points is None :
        If the number of points provided to this function is 4, it automatically assumes the 4 main landmark detector points.
        If the number of points provided is other than 4, it assumes the given num_points points are given in the same order as the csv model points.
        If you want another arrangement, you have to provide both 2d and 3d points yourself.
        The function will shift the 3d points such that nose is at (0,0,0).

    See the docstring of estimate_pose() for the remaining arguments.
   """

    assert camera_type in [OPENCV_FISHEYE, OPENCV_PINHOLE], "Currently, we only support " \
                                                            "opencv_fisheye and opencv_pinhole for the camera_type parameter."
    num_points = len(n_2d_image_space_points) / 2

    if n_3d_local_space_points is None:
        if num_points == 4:
            nose_3d = OUR_3D_MODEL[right_eye_left_eye_nose_chin_name_to_index["nose"]]
            right_eye_3d = OUR_3D_MODEL[right_eye_left_eye_nose_chin_name_to_index["right_eye"]]
            left_eye_3d = OUR_3D_MODEL[right_eye_left_eye_nose_chin_name_to_index["left_eye"]]
            chin_3d = OUR_3D_MODEL[right_eye_left_eye_nose_chin_name_to_index["chin"]]

            eye_center = np.mean([right_eye_3d, left_eye_3d], axis=0)
            # Set 3D model origin to eye center
            model_points_arranged_3d = np.array([
                tuple(np.subtract(right_eye_3d, eye_center)), tuple(np.subtract(left_eye_3d, eye_center)),
                tuple(np.subtract(nose_3d, eye_center)),  tuple(np.subtract(chin_3d, eye_center))
            ])
        else:
            ourModel_centered = OUR_3D_MODEL - np.array(OUR_3D_MODEL[right_eye_left_eye_nose_chin_name_to_index["nose"]])
            model_points_arranged_3d = ourModel_centered[0:num_points].reshape(int(num_points), 1, 3)
    else:
        model_points_arranged_3d = np.array(n_3d_local_space_points).reshape(int(num_points), 1, 3)

    # It must be in this exact dimension (N,1,2) if we use SOLVEPNP_AP3P.
    image_points_arranged = np.array(n_2d_image_space_points).reshape(int(num_points), 1, 2)

    camera_matrix = adjust_camera_matrix(camera_matrix=camera_matrix,
                                         calibration_image_resolution=calibration_image_resolution,
                                         current_image_resolution=current_image_resolution)
    if dist_coeffs is None:
        if camera_type == OPENCV_PINHOLE:
            dist_coeffs = np.zeros((5, 1))
        elif camera_type == OPENCV_FISHEYE:
            dist_coeffs = np.zeros((4, 1))

    return estimate_pose(n_3d_local_space_points=model_points_arranged_3d, n_2d_image_space_points=image_points_arranged,
                         camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, camera_type=camera_type)


def estimate_pose(n_3d_local_space_points, n_2d_image_space_points, camera_matrix, dist_coeffs, camera_type):
    """
    This function takes two corresponding lists of 3d local space points and 2d image space coordinates, and estimates
    the head pose (translation vector in camera coordinate system, and rotation wrt default rotation in the camera view)
    For more info and examples check :
    https://docs.google.com/document/d/1XBSNc7hsP8lXrfpCVTRBchZcxiaL0MXQeSpayE1-qC8/edit#heading=h.3wbbqz4pfldd
    To do the estimation, the function requires an adjusted camera matrix. Adjustment must be performed if
    the calibration resolution is different than the production/current image resolution.
    Check the function named "adjust_camera_matrix" in :
    experiments/head_position_3d/camera_calibration/calibration_scripts/undistort_utils.py

    Arg:
        n_3d_local_space_points: (numpy array) : A numpy array of 3d landmark points in the 3d model local/model space.
            * 3D Points should be in the form [x0,y0,z0, x1,y1,z1, x2,y2,z2, ......].
            * Shape : (num_points * 3,) or (num_points, 3)
            * If you want output translation to be in mm, those points must be in mm.
                Also the camera calibration pattern must have been in mm.
            * The output translation vector will always describe the camera-space position
                of the point you provided as (0,0,0) in the local space coordinates.
                E.g : If you want the output translation vector to point to the nose position,
                you must shift your 3D local space landmarks such that nose is at (0,0,0).
            * We expect the 3D model to be a right-handed-system model, facing towards the screen,
                with x pointing to the right, y pointing up, and z pointing to the screen.
            * OpenCV will output translation and rotation in a right-handed coordinate system,
                where x points to the right, y points down, and z points away from the screen.

        n_2d_image_space_points: (numpy array) A numpy array of 2d pixel coordinates of landmarks in a specific image.
            * Points must have the same order as n_3d_local_space_points.

        camera_matrix : (numpy array) A 3x3 numpy array having the form [[fx, skew, cx], [0, fy, cy], [0, 0, 1]].
            fx,fy,cx,cy must be adjusted depending on the relation between
            calibration resolution and current image resolution.
            * Use the function named "adjust_camera_matrix" in :
                experiments/head_position_3d/camera_calibration/calibration_scripts/undistort_utils.py
                to do the adjustment
            * Check : https://docs.google.com/document/d/1fyXQeuPTZViBkqeeNItsju5zCduIclRfCq84Pi0eH70/
                for more information about the intrinsic camera matrix

        dist_coeffs : (numpy_array) : distortion coefficients from camera calibration process.

        camera_type : (string) Currently, we only support one of "opencv_fisheye" and "opencv_pinhole".
            Note that  different fisheye cameras could need different distortion models.

    Return:
        translation_vector : (numpy array) a vector representing the object position in the camera coordinate frame.
            * The output translation vector will always describe the camera-space position
                of the point you provided as (0,0,0) in the local space coordinates.
            * If you provide the local space points in mm and calibrate the camera in mm, this value will be in mm.

        rotation_vector : Calling cv2.Rodrigues(rotation_vector)[0] will transform that into a 3x3 rotation matrix,
            that describes the rotation of the model in camera space with respect to its default rotation.
            * For more info and examples, check :
            https://docs.google.com/document/d/1XBSNc7hsP8lXrfpCVTRBchZcxiaL0MXQeSpayE1-qC8/edit#heading=h.3wbbqz4pfldd
   """

    assert camera_type in [OPENCV_FISHEYE, OPENCV_PINHOLE], "Currently, we only support " \
                                                            "opencv_fisheye and opencv_pinhole for the camera_type parameter."

    n_2d_image_space_points = np.array(n_2d_image_space_points)
    n_3d_local_space_points = np.array(n_3d_local_space_points)
    num_points_2d = n_2d_image_space_points.size / 2
    num_points_3d = n_3d_local_space_points.size / 3

    assert num_points_2d == num_points_3d, "Number of 3d and 2d points does not match. Given  %d and %d" % (num_points_3d, num_points_2d)

    num_points = num_points_2d
    model_points_arranged_3d = n_3d_local_space_points.reshape(int(num_points), 1, 3)
    image_points_arranged = n_2d_image_space_points.reshape(int(num_points), 1, 2)

    if camera_type == OPENCV_PINHOLE:
        assert dist_coeffs.size == 5, " For opencv_pinhole camera_type, 5 distortion coefficients are needed"

        # TODo: these are the proper shapes. document this
        # cv2.solvePnPRansac(np.random.random((4, 1, 3)), np.random.random((4, 1, 2)), np.eye(3), np.zeros((5, 1)))
        (success, rotation_vector, translation_vector, _) = cv2.solvePnPRansac(model_points_arranged_3d,
                                                                               image_points_arranged, 
                                                                               camera_matrix,
                                                                               dist_coeffs, 
                                                                               flags=cv2.cv2.SOLVEPNP_AP3P)
    elif camera_type == OPENCV_FISHEYE:
        assert dist_coeffs.size == 4, " For opencv_fisheye camera_type, 4 distortion coefficients are needed"
        # First, undistort the landmark points
        undistorted_image_points_arranged = cv2.fisheye.undistortPoints(
            distorted=image_points_arranged, K=camera_matrix, D=dist_coeffs, P=camera_matrix)
        # Then, solvePnP using distortion coefficients of all zeros.
        # I am using 5 zeros here (and not 4) since, after undistortion, this should effectively be a pinhole image
        (success, rotation_vector, translation_vector, _) = cv2.solvePnPRansac(model_points_arranged_3d,
                                                                               undistorted_image_points_arranged,
                                                                               cameraMatrix=camera_matrix,
                                                                               distCoeffs=np.zeros((5, 1)),
                                                                               flags=cv2.cv2.SOLVEPNP_AP3P)

    return translation_vector, rotation_vector
