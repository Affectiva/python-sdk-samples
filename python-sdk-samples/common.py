# !/usr/bin/env python3.5
import csv
import os
from collections import defaultdict
 
import affvisionpy as af
import cv2 as cv2
import math

# Constants
NOT_A_NUMBER = 'nan'
count = 0
TEXT_SIZE = 0.6
PADDING_FOR_SEPARATOR = 5
THRESHOLD_VALUE_FOR_EMOTIONS = 5
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_FILE_NAME = "default"
DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR"
 
#Argparse Variable Constants
WIDTH = 0
HEIGHT = 1
 
process_last_ts = 0.0
capture_last_ts = 0.0
 
header_row = ['TimeStamp', 'faceId', 'upperLeftX', 'upperLeftY', 'lowerRightX', 'lowerRightY', 'confidence', 'interocular_distance',
        'pitch', 'yaw', 'roll', 'joy', 'anger', 'surprise', 'valence', 'fear', 'sadness', 'disgust', 'neutral', 'contempt', 'smile',
        'brow_raise', 'brow_furrow', 'nose_wrinkle', 'upper_lip_raise', 'mouth_open', 'eye_closure', 'cheek_raise', 'yawn',
        'blink', 'blink_rate', 'eye_widen', 'inner_brow_raise', 'lip_corner_depressor'
        ]
 

class Listener(af.ImageListener):
    """
    Listener class that return metrics for processed frames.
 
    """
    def __init__(self):
        super(Listener, self).__init__()
        
        self.measurements_dict = defaultdict()
        self.expressions_dict = defaultdict()
        self.emotions_dict = defaultdict()
        self.bounding_box_dict = defaultdict()
        self.time_metrics_dict = defaultdict()
        self.num_faces = defaultdict()
 
    def results_updated(self, faces, image):
        global process_last_ts
        timestamp = self.time_metrics_dict['timestamp']
        capture_fps = self.time_metrics_dict['cfps']
        global count
        #avoid div by 0 error on the first frame
        try:
            process_fps = 1000.0 / (image.timestamp() - process_last_ts)
        except:
            process_fps = 0
        print("timestamp:" + str(round(timestamp, 0)), "Frame " + str(count), "cfps: " + str(round(capture_fps, 0)), "pfps: " + str(round(process_fps, 0)))
        count +=1
        process_last_ts = image.timestamp()
        self.faces = faces
        # TODO: probably don't need num_faces .. 
        self.num_faces = faces

        self.clear_all_dictionaries()
        for fid, face in faces.items():
            self.measurements_dict[face.get_id()] = defaultdict()
            self.expressions_dict[face.get_id()] = defaultdict()
            self.emotions_dict[face.get_id()] = defaultdict()

            self.measurements_dict[face.get_id()].update(face.get_measurements())
            self.expressions_dict[face.get_id()].update(face.get_expressions())
            self.emotions_dict[face.get_id()].update(face.get_emotions())
            self.bounding_box_dict[face.get_id()] = [face.get_bounding_box()[0].x,
                                                face.get_bounding_box()[0].y,
                                                face.get_bounding_box()[1].x,
                                                face.get_bounding_box()[1].y,
                                                face.get_confidence()]
 
    def image_captured(self, image):
        global capture_last_ts
        try:
            capture_fps = 1000.0 / (image.timestamp() - capture_last_ts)
        except:
            capture_fps = 0
        self.time_metrics_dict['cfps'] = capture_fps
        capture_last_ts = image.timestamp()

    def clear_all_dictionaries(self):
        """
        Clears the dictionary values
        """
        self.measurements_dict.clear()
        self.expressions_dict.clear()
        self.emotions_dict.clear()
        self.bounding_box_dict.clear()
 
 
 
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
       key: str
           Name of the measurement.
       val: str
           Value of the measurement.
       upper_left_y: int
           the upper_left_y co-ordinate of the bounding box
       frame: affvisionpy.Frame
           Frame object to write the measurement on
       x1: upper_left_x co-ordinate of the bounding box whose measurements need to be written
 
    """
    key = str(key)
    padding = 20
    key_name = key.split(".")[1]
    key_text_width, key_text_height = get_text_size(key_name, cv2.FONT_HERSHEY_SIMPLEX, 1)
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
 
 
 
def display_emotions_on_screen(key, val, upper_left_y, frame, x1):
    """
    Display the emotion metrics on screen.
 
        Parameters
        ----------
        key: str
            Name of the emotion.
        val: str
            Value of the emotion.
        upper_left_y: int
            the upper_left_y co-ordinate of the bounding box
        frame: affvisionpy.Frame
            Frame object to write the measurement on
        x1: upper_left_x co-ordinate of the bounding box whose measurements need to be written
 
    """
    key = str(key)
    key_name = key.split(".")[1]
    key_text_width, key_text_height = get_text_size(key_name, cv2.FONT_HERSHEY_SIMPLEX, 1)
 
    val_rect_width = 120
    key_val_width = key_text_width + val_rect_width
    cv2.putText(frame, key_name + ": ", (abs(x1 - key_val_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (0, 0,0), 4, cv2.LINE_AA)
    cv2.putText(frame, key_name + ": ", (abs(x1 - key_val_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (255, 255, 255), 2, cv2.LINE_AA)
    overlay = frame.copy()
 
    if math.isnan(val):
        val = 0
 
    start_box_point_x = abs(x1 - val_rect_width)
    width = 8
    height = 10
 
    rounded_val = roundup(val)
    rounded_val /= 10
    rounded_val = abs(int(rounded_val))
 
    for i in range(0, rounded_val):
        start_box_point_x += 10
        cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                      (start_box_point_x + width, upper_left_y - height), (186, 186, 186), -1)
        if ('valence' in key and val < 0) or ('anger' in key and val > 0):
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
 
 
 
def display_expressions_on_screen(key, val, upper_right_x, upper_right_y, frame, upper_left_y):
    """
    Display the expressions metrics on screen.
 
        Parameters
        ----------
        key: str
            Name of the emotion.
        val: str
            Value of the emotion.
        upper_right_x: int
            the upper_left_x co-ordinate of the bounding box
        upper_right_y: int
            the upper_left_y co-ordinate of the bounding box
        frame: affvisionpy.Frame
            Frame object to write the measurement on
        upper_left_y: upper_left_y co-ordinate of the bounding box whose measurements need to be written
 
    """
    key = str(key)
 
    key_name = key.split(".")[1]
    val_rect_width = 120
    overlay = frame.copy()
    if math.isnan(val):
        val = 0
 
    if 'blink' not in key:
        start_box_point_x = upper_right_x
        width = 8
        height = 10
 
        rounded_val = roundup(val)
        rounded_val /= 10
        rounded_val = int(rounded_val)
        for i in range(0, rounded_val ):
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
        upper_left_y += 25
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
 
 
 
def write_metrics(frame, listener_metrics):
    """
    write measurements, emotions, expressions on screen
 
        Parameters
        ----------
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
        box_height = lower_right_y - upper_left_y
        box_width = lower_right_x - upper_left_x
        upper_right_x = upper_left_x + box_width
        upper_right_y = upper_left_y
 
        for key, val in measurements.items():
            display_measurements_on_screen(key, val, upper_left_y, frame, upper_left_x)
 
            upper_left_y += 25
 
        for key, val in emotions.items():
            display_emotions_on_screen(key, val, upper_left_y, frame, upper_left_x)
            upper_left_y += 25
 
        for key, val in expressions.items():
            display_expressions_on_screen(key, val, upper_right_x, upper_right_y, frame, upper_left_y)
 
            upper_right_y += 25
 


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
    logo = cv2.imread(os.path.dirname(os.path.abspath(__file__))+"/Final logo - RGB Magenta.png")
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
 
 
 
def write_metrics_to_csv_data_list(csv_data, timestamp, listener_metrics):
    """
    Write metrics per frame to a list
 
        Parameters
        ----------
        csv_data:
          list of per frame values to write to
        timestamp: int
           timestamp of each frame
        listener_metrics: dict
            dictionary of dictionaries, gives current listener state
 
    """
    global header_row
    if not listener_metrics["measurements"].keys():
        current_frame_data = {}
        current_frame_data["TimeStamp"] = timestamp
        for field in header_row[1:]:
            current_frame_data[field] = NOT_A_NUMBER
        csv_data.append(current_frame_data)
    else:
        for fid in listener_metrics["measurements"].keys():
            current_frame_data = {}
            current_frame_data["TimeStamp"] = timestamp
            current_frame_data["faceId"] = fid
            upperLeftX, upperLeftY, lowerRightX, lowerRightY = get_bounding_box_points(fid, listener_metrics["bounding_box"])
            current_frame_data["upperLeftX"] = upperLeftX
            current_frame_data["upperLeftY"] = upperLeftY
            current_frame_data["lowerRightX"] = lowerRightX
            current_frame_data["lowerRightY"] = lowerRightY
            for key,val in listener_metrics["measurements"][fid].items():
                current_frame_data[str(key).split('.')[1]] = round(val,4)
            for key,val in listener_metrics["emotions"][fid].items():
                current_frame_data[str(key).split('.')[1]] = round(val,4)
            for key,val in listener_metrics["expressions"][fid].items():
                current_frame_data[str(key).split('.')[1]] = round(val,4)
            current_frame_data["confidence"] = round(listener_metrics["bounding_box"][fid][4],4)
            csv_data.append(current_frame_data)
 
def write_csv_data_to_file(csv_data, csv_file):
    """
    Place logo on the screen
 
        Parameters
        ----------
        csv_data: list
           list to write the data from
        csv_file: list
           file to be written to
    """
    global header_row
    if ".csv" not in csv_file:
        csv_file = csv_file + ".csv"
    with open(csv_file, 'w') as c_file:
        keys = csv_data[0].keys()
        writer = csv.DictWriter(c_file, fieldnames=header_row)
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
 
    c_file.close()
 