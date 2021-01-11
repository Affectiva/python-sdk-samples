# !/usr/bin/env python3
import cv2
import os
import math
import numpy as np
import affvisionpy as af

from body_listener import EDGES, COLORS
from pnp_pose_estimator import estimate_pose_wrapper, get_default_camera_matrix, get_default_distortion_coefficients

TEXT_SIZE = 0.6
PADDING_FOR_SEPARATOR = 5
LEFT_METRIC_OFFSET = 105
LINE_HEIGHT = 25

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

def draw_metrics(frame, listener_metrics, identity_names_dict):
    """
    write metrics on screen

        Parameters
        ----------
        args: parsed command line arguments

        frame: numpy array
            frame to write the metrics on

        listener_metrics: dict
            dictionary of dictionaries, gives current listener state

    """
    for fid in listener_metrics["expressions"].keys():
        expressions = listener_metrics["expressions"][fid]
        emotions = listener_metrics["emotions"][fid]
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, listener_metrics["bounding_box"])

        box_width = lower_right_x - upper_left_x
        upper_right_x = upper_left_x + box_width
        upper_right_y = upper_left_y

        # can be used to show yaw pitch roll
        # for key, val in measurements.items():
        #     display_measurements(key.name, val, upper_left_y, frame, upper_left_x)
        #     upper_left_y += LINE_HEIGHT

        for key, val in emotions.items():
            try:
                key_name = key.name
            except AttributeError:
                key_name = key

            if key_name == 'joy' or key_name == 'anger' or key_name == 'surprise' or key_name == 'neutral':
                display_left_metric(key_name, val, upper_left_x, upper_left_y, frame)
                upper_left_y += LINE_HEIGHT

        # display_gaze(frame, gaze_metric.gaze_region.name, upper_left_x, upper_left_y)
        # upper_left_y += LINE_HEIGHT
        #
        # display_left_metric("gaze_confidence", gaze_metric.confidence, upper_left_x, upper_left_y, frame)
        # upper_left_y += LINE_HEIGHT
        #
        # display_left_metric("glasses", glasses, upper_left_x, upper_left_y, frame)
        # upper_left_y += LINE_HEIGHT
        #
        # if "drowsiness" in listener_metrics:
        #     display_drowsiness(frame, drowsiness_metric, upper_left_x, upper_left_y)
        #     upper_left_y += LINE_HEIGHT
        #
        #     display_left_metric("drowsiness confidence", drowsiness_metric.confidence, upper_left_x, upper_left_y, frame)
        #     upper_left_y += LINE_HEIGHT

        for key, val in expressions.items():
            try:
                key_name = key.name
            except AttributeError:
                key_name = key

            if ("blink_rate" in key_name) or ("blink" in key_name):
                continue
            
            display_expression(key_name, val, upper_right_x, upper_right_y, frame)
            upper_right_y += LINE_HEIGHT

        # draw_age(frame, age_metric, age_category, upper_right_x, upper_right_y)

def draw_age(frame, age_metric, age_category, upper_right_x, upper_right_y):
    age = round(age_metric.age)
    age = 'unknown' if age == -1 else age
    age_confidence = age_metric.confidence
    age_confidence = 0 if math.isnan(age_confidence) else round(age_confidence)

    draw_outlined_text(frame, str(age), upper_right_x + 10, upper_right_y)
    draw_outlined_text(frame, " :age", upper_right_x + LEFT_METRIC_OFFSET, upper_right_y)
    upper_right_y += LINE_HEIGHT

    draw_metric_rects(frame, "age_confidence", age_confidence, upper_right_x + 10, upper_right_y)
    draw_outlined_text(frame, " :age_confidence", upper_right_x + LEFT_METRIC_OFFSET, upper_right_y)
    upper_right_y += LINE_HEIGHT

    draw_outlined_text(frame, age_category, upper_right_x + 10, upper_right_y)
    draw_outlined_text(frame, " :age_category", upper_right_x + LEFT_METRIC_OFFSET, upper_right_y)
    upper_right_y += LINE_HEIGHT

def draw_outlined_text(frame, text, x1, y1, inner_color=(255,255,255)):
    """
    Draw outlined text.

        Parameters
        ----------
        frame: numpy array
            Frame to write the text on
        text: string
            Text to write
        x1: int
            Upper_left_x co-ordinate at which we start drawing the text
        y1: int
            Upper_left_y co-ordinate at which we start drawing the text
    """
    cv2.putText(frame, text, (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX,
                TEXT_SIZE,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX,
                TEXT_SIZE,
                inner_color, 1, cv2.LINE_AA)

def draw_metric_rects(frame, metric_key, metric_val, x1, y1):
    """
    Draw the metric value indicator for the given metric.

        Parameters
        ----------
        frame: numpy array
            Frame to write the metric on
        metric_key: string
            Name of the metric
        metric_val: int
            Value of the metric
        x1: int
            Upper_left_x co-ordinate at which we start drawing the rectangles
        y1: int
            Upper_left_y co-ordinate at which we start drawing the rectangles
    """
    rect_width = 8
    rect_height = 10
    rect_padding = 2
    num_rects = 10

    rounded_val = round(abs(metric_val) / 10)
    for i in range(0, num_rects):
        c = (186, 186, 186)

        if i < rounded_val:
            if ('valence' in metric_key and metric_val < 0) or ('anger' in metric_key and metric_val > 0):
                c = (0, 0, 255)
            else:
                c = (0, 204, 102)

        cv2.rectangle(frame, (x1, y1), (x1 + rect_width, y1 - rect_height), c, -1)

        x1 += 10

def draw_objects(frame, listener_metrics):
    """
    draw objects with its bounding box on screen

        Parameters
        ----------
        frame: numpy array
            frame to write the metrics on

        listener_metrics: dict
            dictionary of dictionaries, gives current listener state

    """
    if "object_type" in listener_metrics:
        for oid in listener_metrics["object_type"].keys():
            upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(oid, listener_metrics[
                "bounding_box"])

            obj_type = listener_metrics["object_type"][oid].name
            # default color == GRAY
            color = (128, 128, 128)
            if "phone" in obj_type:
                # phone color == YELLOW
                color = (0, 255, 255)
            elif "child_seat" in obj_type:
                # child seat color == RED
                color = (0, 0, 255)

            cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), color, 3)

            # extra padding
            upper_left_y -= LINE_HEIGHT

            display_top_metrics("type", obj_type, upper_left_x, upper_left_y, frame)
            upper_left_y -= LINE_HEIGHT

def draw_occupants(frame, listener_metrics):
    """
    draw occupants with its bounding box on screen

        Parameters
        ----------
        frame: numpy array
            frame to write the metrics on

        listener_metrics: dict
            dictionary of dictionaries, gives current listener state

    """
    if "bounding_box" in listener_metrics:
        for occid in listener_metrics["bounding_box"].keys():
            upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(occid, listener_metrics[
                "bounding_box"])

            cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (199, 110, 255), 3)

            if listener_metrics["region_id"][occid] != -1:
                draw_polygon(listener_metrics["region"][occid], frame, (255, 255, 255))

            # extra padding
            upper_left_y -= LINE_HEIGHT

            display_top_metrics("region_confidence", listener_metrics["confidence"][occid], upper_left_x, upper_left_y, frame)
            upper_left_y -= LINE_HEIGHT

            region_type = listener_metrics["region_type"][occid]
            region_id = str(listener_metrics["region_id"][occid])
            display_top_metrics("region_id " + region_id, region_type, upper_left_x, upper_left_y, frame)
            upper_left_y -= LINE_HEIGHT
            display_top_metrics("occupant_id", occid, upper_left_x, upper_left_y, frame)
            upper_left_y -= LINE_HEIGHT
            face_id = listener_metrics["face_id"][occid]
            face_id = 'n/a' if face_id == 'nan' else face_id
            display_top_metrics("face_id", face_id, upper_left_x, upper_left_y, frame)
            upper_left_y -= LINE_HEIGHT
            body_id = listener_metrics["body_id"][occid]
            body_id = 'n/a' if body_id == 'nan' else body_id
            display_top_metrics("body_id", body_id, upper_left_x, upper_left_y, frame)
            upper_left_y -= LINE_HEIGHT
            draw_bodies(frame, listener_metrics)

def draw_bodies(frame, listener_metrics):
    """
    draw bodies points with edges on screen

        Parameters
        ----------
        frame: numpy array
            frame to write the metrics on

        listener_metrics: dict
            dictionary of dictionaries, gives current listener state

    """

    if "body_points" in listener_metrics:
        for body_point in listener_metrics["body_points"].values():
            body_point_keys = body_point.keys()
            for edge_order, edge in enumerate(EDGES):
                if edge[0] in body_point_keys and edge[1] in body_point_keys:
                    start = body_point[edge[0]]
                    end = body_point[edge[1]]
                    cv2.line(frame, (start[0], start[1]), (end[0], end[1]), COLORS[edge_order], 3)

def display_top_metrics(key_name, val, upper_left_x, upper_left_y, frame):
    """
    Display metrics on top of bounding box.

        Parameters
        ----------
        key_name: string
            Name of the metrics.
        val: float
            Value of the expression.
        upper_left_x: int
            the upper_left_x co-ordinate of the bounding box
        upper_left_y: int
            the upper_left_y co-ordinate of the bounding box
        frame: numpy array
            Frame object to write the expression on

    """
    key_text = key_name + ": "
    text_width, key_text_height = get_text_size(key_text, cv2.FONT_HERSHEY_DUPLEX, 1)

    draw_outlined_text(frame, key_text, upper_left_x, upper_left_y)

    if 'region_confidence' in key_name or 'object_confidence' in key_name:
        if math.isnan(val):
            val = 0
        draw_metric_rects(frame, key_name, val, upper_left_x + text_width, upper_left_y)
    else:
        draw_outlined_text(frame, str(val), upper_left_x + text_width, upper_left_y)

def draw_polygon(points, frame, color):
    pts = []
    for pt in points:
        pts.append([int(pt.x), int(pt.y)])
    cv2.polylines(frame, [np.array(pts)], True, color, 3)

def get_bounding_box_points(fid, bounding_box_dict):
    """
    Fetch upper_left_x, upper_left_y, lower_right_x, lwoer_right_y points of the bounding box.

        Parameters
        ----------
        fid: int
            face id of the face to get the bounding box for

        bounding_box_dict: int -> list of float
            dictionary from face id to array of bbox points

        Returns
        -------
        tuple of int values
            tuple with upper_left_x, upper_left_y, upper_right_x, upper_right_y values
    """
    return (int(bounding_box_dict[fid][0]),
            int(bounding_box_dict[fid][1]),
            int(bounding_box_dict[fid][2]),
            int(bounding_box_dict[fid][3]))

def get_face_landmark_points(fid, face_landmark_points_dict):
    """
    Fetch landmark points from given dictionary

    Inputs:
    * fid: int
        Face ID to get the landmark points for

    Outputs:
    * landmark_points: Array[float]
        outer_right_eye, outer_left_eye, nose_tip, chin_tip
    """

    return [face_landmark_points_dict[fid][af.FacePoint.outer_right_eye],
    face_landmark_points_dict[fid][af.FacePoint.outer_left_eye],
    face_landmark_points_dict[fid][af.FacePoint.nose_tip],
    face_landmark_points_dict[fid][af.FacePoint.chin_tip]]

def draw_bounding_box(frame, listener_metrics, show_emotion):
    """
    For each frame, draw the bounding box on screen.

        Parameters
        ----------
        frame: numpy array
            Frame object to draw the bounding box on.

        listener_metrics: dict
            dictionary of dictionaries, gives current listener state

    """
    if show_emotion:
        emotion_value_threshold = 5
        for fid in listener_metrics["bounding_box"].keys():
            upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, listener_metrics["bounding_box"])
            if fid in listener_metrics["emotions"]:
                for key in listener_metrics["emotions"][fid]:
                    if 'valence' in str(key):
                        valence_value = listener_metrics["emotions"][fid][key]
                    if 'anger' in str(key):
                        anger_value = listener_metrics["emotions"][fid][key]
                    if 'joy' in str(key):
                        joy_value = listener_metrics["emotions"][fid][key]
                
                if valence_value < 0 and anger_value >= emotion_value_threshold:
                    cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (0, 0, 255), 3)
                elif valence_value >= emotion_value_threshold and joy_value >= emotion_value_threshold:
                    cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (0, 255, 0), 3)
                else:
                    cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (21, 169, 167), 3)
    else:
        for fid in listener_metrics["bounding_box"].keys():
            upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, listener_metrics["bounding_box"])
            cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (21, 169, 167), 3)

def draw_and_calculate_3d_pose(frame, camera_matrix, camera_type, dist_coefficients, listener_metrics):
    """
    Calculate 3d pose from landmark face points, then draw arrows indicating 3d pose
    on the given frame.

    """
    axis_length = 50

    if "face_landmark_pts" in listener_metrics:
        face_landmark_points_dict = listener_metrics["face_landmark_pts"]
        for fid in face_landmark_points_dict.keys():
            landmark_pts = get_face_landmark_points(fid, face_landmark_points_dict)
            np_landmark_pts = np.array([
                [landmark_pts[0].x], [landmark_pts[0].y],
                [landmark_pts[1].x], [landmark_pts[1].y],
                [landmark_pts[2].x], [landmark_pts[2].y],
                [landmark_pts[3].x], [landmark_pts[3].y]
            ], dtype=np.float)

            current_image_resolution = calibration_image_resolution = frame.shape[:2]

            if (camera_matrix is None):
                camera_matrix = get_default_camera_matrix(calibration_image_resolution=calibration_image_resolution,
                                                          current_image_resolution=current_image_resolution)

            if (dist_coefficients is None):
                dist_coefficients = get_default_distortion_coefficients(camera_type)

            (translation, rotation) = estimate_pose_wrapper(
                np_landmark_pts,
                camera_matrix,
                current_image_resolution,
                calibration_image_resolution,
                camera_type,
                dist_coeffs=dist_coefficients
            )

            (nosecenter_2d, _) = cv2.projectPoints((0, 0, 0),
                                                   rotation,
                                                   translation,
                                                   camera_matrix,
                                                   dist_coefficients)

            (arrow_pts, _) = cv2.projectPoints(np.array([[axis_length, 0.0, 0.0],
                                                         [0, axis_length, 0],
                                                         [0, 0, axis_length]]),
                                               rotation,
                                               translation,
                                               camera_matrix,
                                               dist_coefficients)

            nosecenter_origin = tuple(map(int, tuple(nosecenter_2d[0].ravel())))
            cv2.line(frame, nosecenter_origin, tuple(map(int, tuple(arrow_pts[0].ravel()))), (255, 0, 0), 2) #x blue
            cv2.line(frame, nosecenter_origin, tuple(map(int, tuple(arrow_pts[1].ravel()))), (0, 255, 0), 2) #y green
            cv2.line(frame, nosecenter_origin, tuple(map(int, tuple(arrow_pts[2].ravel()))), (0, 0, 255), 2) #z red


def get_text_size(text, font, thickness):
    """
    Get the size occupied by a particular text string

       Parameters
       ----------
       text: str
           The text string to find size of.
       font: str
           font size of the text string
       thickness: int
           thickness of the font

       Returns
       -------
       tuple of int values
           text width, text height
    """
    text_size = cv2.getTextSize(text, font, TEXT_SIZE, thickness)
    return text_size[0][0], text_size[0][1]

def display_gaze(frame, gaze_region_name, upper_left_x, upper_left_y):
    """
    Display gaze on screen to the left of the bounding box

        Parameters
        ----------
        frame: numpy array
            Frame to write the gaze metric on
        gaze_region_name: string
            Value of the gaze metric
        upper_left_x: int
            the upper_left_x co-ordinate of the bounding box
        upper_left_y: int
            the upper_left_y co-ordinate of the bounding box
    """
    gaze_text_size = 132  # this was precalculated
    draw_outlined_text(frame, "gaze_region: ", abs(upper_left_x - gaze_text_size - LEFT_METRIC_OFFSET), upper_left_y)
    draw_outlined_text(frame, gaze_region_name, abs(upper_left_x - LEFT_METRIC_OFFSET), upper_left_y)

def display_distraction(frame, eyes_on_road, upper_left_x, upper_left_y):
    if eyes_on_road:
        distraction_text = "on road"
        distraction_text_color = (150,210,50)
    else:
        distraction_text = "off road"
        distraction_text_color = (51, 87, 255)

    text_size = 54
    draw_outlined_text(frame, "eyes: ", abs(upper_left_x - text_size - LEFT_METRIC_OFFSET), upper_left_y)
    draw_outlined_text(frame, distraction_text, abs(upper_left_x - LEFT_METRIC_OFFSET), upper_left_y, distraction_text_color)

def display_drowsiness(frame, drowsiness_metric, upper_left_x, upper_left_y):
    """
    Display drowsiness metrics on screen to the left of the bounding box

        Parameters
        ----------
        frame: numpy array
            Frame to write the gaze metric on
        drowsiness_metric: DrowsinessMetric
            Value of the drowsiness metric
        upper_left_x: int
            the upper_left_x co-ordinate of the bounding box
        upper_left_y: int
            the upper_left_y co-ordinate of the bounding box
    """

    key_name = "drowsiness level: "
    key_text_width, key_text_height = get_text_size(key_name, cv2.FONT_HERSHEY_SIMPLEX, 1)
    draw_outlined_text(frame, key_name, abs(upper_left_x - key_text_width - LEFT_METRIC_OFFSET), upper_left_y)

    text_color = (255,255,255)
    val_text = drowsiness_metric.drowsiness.name
    if val_text == "asleep":
        text_color = (51, 87, 255)
    elif val_text == "severe":
        text_color = (90, 160, 250)
    elif val_text == "moderate":
        text_color = (0, 204, 255)
        
    draw_outlined_text(frame, drowsiness_metric.drowsiness.name, abs(upper_left_x - LEFT_METRIC_OFFSET), upper_left_y, text_color)

def display_measurements(key_name, val, upper_left_y, frame, x1):
    """
    Display the measurement metrics on screen.

       Parameters
       ----------
       key_name: string
           Name of the measurement.
       val: float
           Value of the measurement.
       upper_left_y: int
           the upper_left_y co-ordinate of the bounding box
       frame: numpy array
           Frame object to write the measurement on
       x1: int
           upper_left_x co-ordinate of the bounding box whose measurements need to be written

    """
    key_text_width, key_text_height = get_text_size(key_name, cv2.FONT_HERSHEY_SIMPLEX, 1)

    if math.isnan(val):
        val = 0
    val_text = str(round(val, 2))
    val_text_width, val_text_height = get_text_size(val_text, cv2.FONT_HERSHEY_SIMPLEX, 1)
    max_val_text_width = 83
    key_val_width = key_text_width + max_val_text_width

    draw_outlined_text(frame, key_name + ": ", abs(x1 - key_val_width - PADDING_FOR_SEPARATOR), upper_left_y)
    draw_outlined_text(frame, val_text, abs(x1 - val_text_width), upper_left_y)

def display_left_metric(key_name, val, upper_left_x, upper_left_y, frame):
    """
    Display metrics on screen to the left of bounding box.

        Parameters
        ----------
        key_name: string
            Name of the metric.
        val: float
            Value of the metric.
        upper_left_x: int
            the upper_left_x co-ordinate of the bounding box
        upper_left_y: int
            the upper_left_y co-ordinate of the bounding box
        frame: numpy array
            Frame object to write the metric on

    """
    key_text = key_name + ": "
    text_width, key_text_height = get_text_size(key_text, cv2.FONT_HERSHEY_DUPLEX, 1)

    total_rect_width = LEFT_METRIC_OFFSET

    key_val_width = text_width + total_rect_width
    draw_outlined_text(frame, key_text, abs(upper_left_x - key_val_width), upper_left_y)

    if math.isnan(val):
        val = 0

    draw_metric_rects(frame, key_name, val, abs(upper_left_x - total_rect_width), upper_left_y)

def display_expression(key_name, val, upper_right_x, upper_right_y, frame):
    """
    Display an expression metric on screen, showing the name next to a horizontal segmented bar representing the value

        Parameters
        ----------
        key: string
            Name of the expression.
        val: float
            Value of the expression.
        upper_right_x: int
            the upper_right_x co-ordinate of the bounding box
        upper_right_y: int
            the upper_right_y co-ordinate of the bounding box
        frame: numpy array
            Frame object to write the expression on

    """
    val_rect_width = 120
    if math.isnan(val):
        val = 0

    if 'blink' not in key_name:
        draw_metric_rects(frame, key_name, val, upper_right_x + 10, upper_right_y)

    else:
        draw_outlined_text(frame, str(val), upper_right_x + 10, upper_right_y)

    draw_outlined_text(frame, " :" + key_name, upper_right_x + LEFT_METRIC_OFFSET, upper_right_y)

def display_identity(frame, identity, upper_left_y, upper_left_x, identity_names_dict):
    """
        Display the face identity metrics on screen.

            Parameters
            ----------
            frame: numpy array
                Frame object to write the measurement on
            identity: int
                identity of the occupant in the current frame
            upper_left_y: upper left Y coordinate of the face bounding box
            upper_left_x:  upper left X coordinate of the face bounding box

        """

    upper_left_x += 25

    if str(identity) in identity_names_dict:
        name = identity_names_dict[str(identity)]
        id_name = "Identity " + str(identity) + ": " + name
    else:
        name = "Unknown"
        id_name = "Identity " + str(identity) + ": " + name

    cv2.putText(frame, id_name, (upper_left_x, upper_left_y - 10), cv2.FONT_HERSHEY_DUPLEX, TEXT_SIZE, (255, 255, 255),
                1, cv2.LINE_AA)

def get_affectiva_logo(frame_width, frame_height):
    """
    Return the properly sized logo for the given sized frame

    Parameters
    ----------
    frame_width: int
        width of the frame the logo will be placed in
    frame_height: int
        height of the frame the logo will be placed in
    """
    logo = cv2.imread(IMAGES_DIR + "/Final logo - RGB Magenta.png")
    logo_width = int(frame_width / 3)
    logo_height = int(frame_height / 10)
    logo = cv2.resize(logo, (logo_width, logo_height))
    return logo

def draw_affectiva_logo(frame, logo, width, height):
    """
    Place logo on the screen

        Parameters
        ----------
        frame: numpy array
           Frame to place the logo on
        logo: numpy array
            Logo to place in the frame
        width: int
           width of the frame
        height: int
           height of the frame
    """
    logo_height, logo_width = logo.shape[:2]

    y1, y2 = 0, logo_height
    x1, x2 = width - logo_width, width
    # Remove the white background from the logo so that only the word "Affectiva" is visible on screen
    for c in range(0, 3):
        alpha = logo[0:logo_height, 0:logo_width, 1] / 255.0
        color = logo[0:logo_height, 0:logo_width, c] * (1.0 - alpha)
        beta = frame[y1:y2, x1:x2, c] * (alpha)
        frame[y1:y2, x1:x2, c] = color + beta

def draw_gaze_region(frame, gaze_metric):
    idx = int(gaze_metric.gaze_region)

    height = frame.shape[0]

    ypadding = 10
    xpadding = 10
    if idx == -1:
        idx = 0

    img_name = "gaze_region_{0}.png".format(idx)
    img = cv2.imread(os.path.join(IMAGES_DIR, img_name), cv2.IMREAD_UNCHANGED)

    img_height = int(height / 4)
    img_width = int(img.shape[1] * float(img_height) / img.shape[0])
    img = cv2.resize(img, (img_width, img_height))

    y1, y2 = height - img_height - ypadding, height - ypadding
    x1, x2 = xpadding, img_width + xpadding

    alpha = img[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha

    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha * img[:, :, c] + alpha_frame * frame[y1:y2, x1:x2, c])

def check_bounding_box_outside(width, height, bounding_box_dict):
    """
    Check if bounding box values are going outside the screen in case of face going outside

        Parameters
        ----------
        width: int
           width of the frame
        height: int
           height of the frame
        bounding_box_dict: int -> list of float
            dictionary from face id to array of bbox points

    Returns
    -------
    boolean: indicating if the bounding box is outside the frame or not
    """
    for fid in bounding_box_dict.keys():
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, bounding_box_dict)
        if upper_left_x < 0 or lower_right_x > width or upper_left_y < 0 or lower_right_y > height:
            return True
        return False
