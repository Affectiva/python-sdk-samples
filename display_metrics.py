# !/usr/bin/env python3
import cv2
import os
import math
import numpy as np

TEXT_SIZE = 0.6
PADDING_FOR_SEPARATOR = 5
LEFT_METRIC_OFFSET = 105
LINE_HEIGHT = 25

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

def draw_metrics(frame, listener_metrics):
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
    for fid in listener_metrics["measurements"].keys():
        measurements = listener_metrics["measurements"][fid]
        expressions = listener_metrics["expressions"][fid]
        emotions = listener_metrics["emotions"][fid]
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, listener_metrics["bounding_box"])

        box_width = lower_right_x - upper_left_x
        upper_right_x = upper_left_x + box_width
        upper_right_y = upper_left_y
        
        for key, val in measurements.items():
            display_measurements(key.name, val, upper_left_y, frame, upper_left_x)
            upper_left_y += LINE_HEIGHT

        for key, val in emotions.items():
            display_left_metric(key.name, val, upper_left_x, upper_left_y, frame)
            upper_left_y += LINE_HEIGHT

        for key, val in expressions.items():
            display_expression(key.name, val, upper_right_x, upper_right_y, frame)
            upper_right_y += LINE_HEIGHT


def draw_outlined_text(frame, text, x1, y1):
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
                (255, 255, 255), 1, cv2.LINE_AA)

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

def draw_bounding_box(frame, listener_metrics):
    """
    For each frame, draw the bounding box on screen.

        Parameters
        ----------
        frame: numpy array
            Frame object to draw the bounding box on.

        listener_metrics: dict
            dictionary of dictionaries, gives current listener state

    """
    emotion_value_threshold = 5
    for fid in listener_metrics["bounding_box"].keys():
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, listener_metrics["bounding_box"])
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
