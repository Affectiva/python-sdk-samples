# !/usr/bin/env python3
import cv2
import os
import math

TEXT_SIZE = 0.6
PADDING_FOR_SEPARATOR = 5
THRESHOLD_VALUE_FOR_EMOTIONS = 5


IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

def draw_metrics(frame, listener_metrics, identity_names_dict):

    """
    write metrics on screen
 
        Parameters
        ----------
        args: parsed command line arguments

        frame: affvisionpy.Frame
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

        if "identities" in listener_metrics:
            print(listener_metrics["identities"][fid])
            display_identity_on_screen(frame, listener_metrics["identities"][fid], upper_left_y, upper_left_x, identity_names_dict)
 
        for key, val in measurements.items():
            display_measurements_on_screen(key, val, upper_left_y, frame, upper_left_x)
            upper_left_y += 25

        for key, val in emotions.items():
            display_left_metrics(key.name, val, upper_left_x, upper_left_y, frame)
            upper_left_y += 25

        gaze_reg = "gaze_region: " + str(listener_metrics["gaze_metric"][fid].gaze_region.name)
        key_text_width, key_text_height = get_text_size(gaze_reg, cv2.FONT_HERSHEY_SIMPLEX, 1)

        cv2.putText(frame, gaze_reg, (abs(upper_left_x - key_text_width - PADDING_FOR_SEPARATOR), upper_left_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    TEXT_SIZE,
                    (255, 255, 255), 2, cv2.LINE_AA)
        upper_left_y += 25

        display_left_metrics("gaze_confidence", listener_metrics["gaze_metric"][fid].confidence, upper_left_x, upper_left_y, frame)
        upper_left_y += 25

        display_left_metrics("glasses", listener_metrics["glasses"][fid], upper_left_x, upper_left_y, frame)
        upper_left_y += 25

        for key, val in expressions.items():
            display_expressions_on_screen(key, val, upper_right_x, upper_right_y, frame)
            upper_right_y += 25
        # draw gaze region for the first face. (ideally we should draw it for the driver only)
        gaze_metrics = listener_metrics["gaze_metric"]
        if len(gaze_metrics):
            metric = next(iter(gaze_metrics.values()))
            draw_gaze_region(frame, metric)

            
def draw_objects(frame, listener_metrics):
    """
    write objects with its bounding box on screen

        Parameters
        ----------
        frame: affvisionpy.Frame
            frame to write the metrics on

        listener_metrics: dict
            dictionary of dictionaries, gives current listener state

    """
    if "object_type" in listener_metrics:
        for oid in listener_metrics["object_type"].keys():
            upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(oid, listener_metrics[
                "bounding_box"])
            box_width = lower_right_x - upper_left_x
            upper_right_x = upper_left_x + box_width
            upper_right_y = upper_left_y

            cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (0, 0, 255), 3)

            obj_type = listener_metrics["object_type"][oid].name
            cv2.putText(frame, obj_type, (upper_right_x, upper_right_y), cv2.FONT_HERSHEY_DUPLEX, TEXT_SIZE,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, obj_type, (upper_right_x, upper_right_y), cv2.FONT_HERSHEY_DUPLEX, TEXT_SIZE,
                        (255, 255, 255), 1, cv2.LINE_AA)


def get_bounding_box_points(fid, bounding_box_dict):
    """
    Fetch upper_left_x, upper_left_y, lower_right_x, lwoer_right_y points of the bounding box.
 
        Parameters
        ----------
        fid: int
            face id of the face to get the bounding box for
 
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
        frame: affvisionPy.Frame
            Frame object to draw the bounding box on.

        listener_metrics: dict
            dictionary of dictionaries, gives current listener state
 
    """
    for fid in listener_metrics["bounding_box"].keys():
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, listener_metrics["bounding_box"])
        for key in listener_metrics["emotions"][fid]:
            if 'valence' in str(key):
                valence_value = listener_metrics["emotions"][fid][key]
            if 'anger' in str(key):
                anger_value = listener_metrics["emotions"][fid][key]
            if 'joy' in str(key):
                joy_value = listener_metrics["emotions"][fid][key]

        if valence_value < 0 and anger_value >= THRESHOLD_VALUE_FOR_EMOTIONS:
            cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (0, 0, 255), 3)
        elif valence_value >= THRESHOLD_VALUE_FOR_EMOTIONS and joy_value >= THRESHOLD_VALUE_FOR_EMOTIONS:
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

def display_measurements_on_screen(key, val, upper_left_y, frame, x1):
    """
    Display the measurement metrics on screen.
 
       Parameters
       ----------
       key: affvisionpy.Measurement
           Name of the measurement.
       val: float
           Value of the measurement.
       upper_left_y: int
           the upper_left_y co-ordinate of the bounding box
       frame: affvisionpy.Frame
           Frame object to write the measurement on
       x1: upper_left_x co-ordinate of the bounding box whose measurements need to be written
 
    """
    key_name = key.name
    key_text_width, key_text_height = get_text_size(key_name, cv2.FONT_HERSHEY_SIMPLEX, 1)

    if math.isnan(val):
        val = 0
    val_text = str(round(val, 2))
    val_text_width, val_text_height = get_text_size(val_text, cv2.FONT_HERSHEY_SIMPLEX, 1)

    key_val_width = key_text_width + val_text_width

    cv2.putText(frame, key_name + ": ", (abs(x1 - key_val_width - PADDING_FOR_SEPARATOR), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (255, 255, 255))
    cv2.putText(frame, val_text, (abs(x1 - val_text_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,
                (255, 255, 255))

def display_left_metrics(key_name, val, upper_left_x, upper_left_y, frame):
    """
    Display metrics on screen to the left of bounding box.
 
        Parameters
        ----------
        key_name: string
            Name of the emotion.
        val: float
            Value of the emotion.
        upper_left_x: int
            the upper_left_x co-ordinate of the bounding box
        upper_left_y: int
            the upper_left_y co-ordinate of the bounding box
        frame: affvisionpy.Frame
            Frame object to write the emotion on
 
    """
    key_text_width, key_text_height = get_text_size(key_name, cv2.FONT_HERSHEY_SIMPLEX, 1)

    val_rect_width = 120
    key_val_width = key_text_width + val_rect_width
    cv2.putText(frame, key_name + ": ", (abs(upper_left_x - key_val_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, key_name + ": ", (abs(upper_left_x - key_val_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (255, 255, 255), 2, cv2.LINE_AA)
    overlay = frame.copy()

    if math.isnan(val):
        val = 0

    start_box_point_x = abs(upper_left_x - val_rect_width)
    width = 8
    height = 10

    rounded_val = roundup(val)
    rounded_val /= 10
    rounded_val = abs(int(rounded_val))

    for i in range(0, rounded_val):
        start_box_point_x += 10
        cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                      (start_box_point_x + width, upper_left_y - height), (186, 186, 186), -1)
        if ('valence' in key_name and val < 0) or ('anger' in key_name and val > 0):
            cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                          (start_box_point_x + width, upper_left_y - height), (0, 0, 255), -1)
        else:
            cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                          (start_box_point_x + width, upper_left_y - height), (0, 204, 102), -1)
    for i in range(rounded_val, 10):
        start_box_point_x += 10
        cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                      (start_box_point_x + width, upper_left_y - height), (186, 186, 186), -1)

    alpha = 0.8
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def display_expressions_on_screen(key, val, upper_right_x, upper_right_y, frame):
    """
    Display the expressions metrics on screen.
 
        Parameters
        ----------
        key: affvisionpy.Expression
            Name of the expression.
        val: float
            Value of the expression.
        upper_right_x: int
            the upper_right_x co-ordinate of the bounding box
        upper_right_y: int
            the upper_right_y co-ordinate of the bounding box
        frame: affvisionpy.Frame
            Frame object to write the expression on
 
    """
    key_name = key.name
    val_rect_width = 120
    overlay = frame.copy()
    if math.isnan(val):
        val = 0

    if 'blink' not in key_name:
        start_box_point_x = upper_right_x
        width = 8
        height = 10

        rounded_val = roundup(val)
        rounded_val /= 10
        rounded_val = int(rounded_val)
        for i in range(0, rounded_val):
            start_box_point_x += 10
            cv2.rectangle(overlay, (start_box_point_x, upper_right_y),
                          (start_box_point_x + width, upper_right_y - height), (186, 186, 186), -1)
            cv2.rectangle(overlay, (start_box_point_x, upper_right_y),
                          (start_box_point_x + width, upper_right_y - height), (0, 204, 102), -1)
        for i in range(rounded_val, 10):
            start_box_point_x += 10
            cv2.rectangle(overlay, (start_box_point_x, upper_right_y),
                          (start_box_point_x + width, upper_right_y - height), (186, 186, 186), -1)

        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        cv2.putText(frame, str(val), (upper_right_x, upper_right_y), cv2.FONT_HERSHEY_DUPLEX, TEXT_SIZE,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, str(val), (upper_right_x, upper_right_y), cv2.FONT_HERSHEY_DUPLEX, TEXT_SIZE,
                    (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, " :" + str(key_name), (upper_right_x + val_rect_width, upper_right_y), cv2.FONT_HERSHEY_DUPLEX,
                TEXT_SIZE,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, " :" + str(key_name), (upper_right_x + val_rect_width, upper_right_y), cv2.FONT_HERSHEY_DUPLEX,
                TEXT_SIZE,
                (255, 255, 255), 1, cv2.LINE_AA)

def display_confidence_on_screen(key, val, upper_left_x, upper_left_y, frame):
    """
    Display the confidence metrics on screen.

        Parameters
        ----------
        key: str
            Name of the confidence.
        val: float
            Value of the emotion.
        upper_left_x: int
            the upper_left_x co-ordinate of the bounding box
        upper_left_y: int
            the upper_left_y co-ordinate of the bounding box
        frame: affvisionpy.Frame
            Frame object to write the confidence on

    """
    key_text_width, key_text_height = get_text_size(key, cv2.FONT_HERSHEY_SIMPLEX, 1)

    val_rect_width = 120
    key_val_width = key_text_width + val_rect_width
    cv2.putText(frame, key + ": ", (abs(upper_left_x - key_val_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, key + ": ", (abs(upper_left_x - key_val_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (255, 255, 255), 2, cv2.LINE_AA)
    overlay = frame.copy()

    if math.isnan(val):
        val = 0

    start_box_point_x = abs(upper_left_x - val_rect_width)
    width = 8
    height = 10

    rounded_val = roundup(val)
    rounded_val /= 10
    rounded_val = abs(int(rounded_val))

    for i in range(0, rounded_val):
        start_box_point_x += 10
        cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                      (start_box_point_x + width, upper_left_y - height), (186, 186, 186), -1)

        cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                      (start_box_point_x + width, upper_left_y - height), (0, 204, 102), -1)

    for i in range(rounded_val, 10):
        start_box_point_x += 10
        cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                      (start_box_point_x + width, upper_left_y - height), (186, 186, 186), -1)

    alpha = 0.8
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def display_identity_on_screen(frame, identity, upper_left_y, upper_left_x, identity_names_dict):
    """
        Display the face identity metrics on screen.

            Parameters
            ----------
            frame: affvisionpy.Frame
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

def draw_affectiva_logo(frame, width, height):
    """
    Place logo on the screen
 
        Parameters
        ----------
        frame: affvisionpy.Frame
           Frame to place the logo on
        width: int
           width of the frame
        height: int
           height of the frame
    """

    logo = cv2.imread(os.path.join(IMAGES_DIR, 'Final logo - RGB Magenta.png'))

    logo_width = int(width / 3)
    logo_height = int(height / 10)
    logo = cv2.resize(logo, (logo_width, logo_height))

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

    # width = frame.shape[1]
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
 
    Returns
    -------
    boolean: indicating if the bounding box is outside the frame or not
    """
    for fid in bounding_box_dict.keys():
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, bounding_box_dict)
        if upper_left_x < 0 or lower_right_x > width or upper_left_y < 0 or lower_right_y > height:
            return True
        return False

def roundup(num):
    """
    Round up the number to the nearest 10.
 
       Parameters
       ----------
       num: int
           number to be rounded up to 10.
 
       Returns
       -------
       int
           Rounded up value of the number to 10
    """
    if (num / 10.0) < 5:
        return int(math.floor(num / 10.0)) * 10
    return int(math.ceil(num / 10.0)) * 10