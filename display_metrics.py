# !/usr/bin/env python3
import cv2
import os
import math
import numpy as np

from body_listener import EDGES, COLORS

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
    # draw gaze region for the first face. (ideally we should draw it for the driver only)
    gaze_metrics = listener_metrics["gaze_metric"]
    if len(gaze_metrics):
        metric = next(iter(gaze_metrics.values()))
        draw_gaze_region(frame, metric)

    for fid in listener_metrics["measurements"].keys():
        measurements = listener_metrics["measurements"][fid]
        expressions = listener_metrics["expressions"][fid]
        emotions = listener_metrics["emotions"][fid]
        gaze_metric = listener_metrics["gaze_metric"][fid]
        glasses = listener_metrics["glasses"][fid]
        age_metric = listener_metrics["age_metric"][fid]
        age_category = listener_metrics["age_category"][fid]
        if "drowsiness" in listener_metrics:
            drowsiness_metric = listener_metrics["drowsiness"][fid]
        if "identities" in listener_metrics:
            identity = listener_metrics["identities"][fid]
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, listener_metrics["bounding_box"])

        box_width = lower_right_x - upper_left_x
        upper_right_x = upper_left_x + box_width
        upper_right_y = upper_left_y

        if "identities" in listener_metrics:
            display_identity(frame, identity, upper_left_y, upper_left_x, identity_names_dict)

        for key, val in measurements.items():
            display_measurements(key.name, val, upper_left_y, frame, upper_left_x)
            upper_left_y += LINE_HEIGHT

        for key, val in emotions.items():
            display_left_metric(key.name, val, upper_left_x, upper_left_y, frame)
            upper_left_y += LINE_HEIGHT

        display_gaze(frame, gaze_metric.gaze_region.name, upper_left_x, upper_left_y)
        upper_left_y += LINE_HEIGHT

        display_left_metric("gaze_confidence", gaze_metric.confidence, upper_left_x, upper_left_y, frame)
        upper_left_y += LINE_HEIGHT

        display_left_metric("glasses", glasses, upper_left_x, upper_left_y, frame)
        upper_left_y += LINE_HEIGHT

        if "drowsiness" in listener_metrics:
            display_drowsiness(frame, drowsiness_metric, upper_left_x, upper_left_y)
            upper_left_y += LINE_HEIGHT

            display_left_metric("drowsiness confidence", drowsiness_metric.confidence, upper_left_x, upper_left_y, frame)
            upper_left_y += LINE_HEIGHT

        for key, val in expressions.items():
            display_expression(key.name, val, upper_right_x, upper_right_y, frame)
            upper_right_y += LINE_HEIGHT

        draw_age(frame, age_metric, age_category, upper_right_x, upper_right_y)

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

            if listener_metrics["region_id"][oid] != -1:
                draw_polygon(listener_metrics["region"][oid], frame, (255, 255, 255))

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

            display_top_metrics("object_confidence", listener_metrics["confidence"][oid], upper_left_x, upper_left_y, frame)
            upper_left_y -= LINE_HEIGHT
            display_top_metrics("region_confidence", listener_metrics["region_confidence"][oid], upper_left_x, upper_left_y, frame)
            upper_left_y -= LINE_HEIGHT

            region_type = listener_metrics["region_type"][oid]
            region_id = str(listener_metrics["region_id"][oid])
            display_top_metrics("region_id " + region_id, region_type, upper_left_x, upper_left_y, frame)
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
    draw_outlined_text(frame, drowsiness_metric.drowsiness.name, abs(upper_left_x - LEFT_METRIC_OFFSET), upper_left_y)

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
