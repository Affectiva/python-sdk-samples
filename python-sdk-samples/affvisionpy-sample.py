# !/usr/bin/env python3.5
import argparse
import csv
import os
import time
from collections import defaultdict
from collections import namedtuple

import affvisionpy as af
import cv2 as cv2
import math
import queue


# Constants
NOT_A_NUMBER = 'NaN'
count = 0
TEXT_SIZE = 0.6
PADDING_FOR_SEPARATOR = 5
THRESHOLD_VALUE_FOR_EMOTIONS = 5
DECIMAL_ROUNDING_FACTOR = 2
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_FILE_NAME = "default"

#Argparse Variable Constants
WIDTH = 0
HEIGHT = 1

process_last_ts = 0.0
capture_last_ts = 0.0

Mood = namedtuple("Mood", ['mood','confidence','dominant_emotion','dominant_emotion_confidence'])
header_row = ['TimeStamp', 'faceId', 'upperLeftX', 'upperLeftY', 'lowerRightX', 'lowerRightY', 'confidence', 'interocular_distance',
        'pitch', 'yaw', 'roll', 'joy', 'anger', 'surprise', 'valence', 'fear', 'sadness', 'disgust', 'neutral', 'smile',
        'brow_raise', 'brow_furrow', 'nose_wrinkle', 'upper_lip_raise', 'mouth_open', 'eye_closure', 'cheek_raise', 'yawn',
        'blink', 'blink_rate', 'eye_widen', 'inner_brow_raise', 'lip_corner_depressor', 'mood', 'dominant_emotion', 'dominant_emotion_confidence'
        ]

measurements_dict = defaultdict()
expressions_dict = defaultdict()
emotions_dict = defaultdict()
bounding_box_dict = defaultdict()
mood_dict = defaultdict()
time_metrics_dict = defaultdict()



class Listener(af.ImageListener):
    """
    Listener class that return metrics for processed frames.

    """
    def __init__(self):
        super(Listener, self).__init__()

    def results_updated(self, faces, image):
        global process_last_ts
        timestamp = time_metrics_dict['timestamp']
        capture_fps = time_metrics_dict['cfps']
        global count
        process_fps = 1000.0 / (image.timestamp() - process_last_ts)
        print("timestamp:" + str(round(timestamp, 0)), "Frame " + str(count), "cfps: " + str(round(capture_fps, 0)), "pfps: " + str(round(process_fps, 0)))
        count +=1
        process_last_ts = image.timestamp()
        self.faces = faces
        global num_faces
        num_faces = faces
        for fid, face in faces.items():
            measurements_dict[face.get_id()] = defaultdict()
            expressions_dict[face.get_id()] = defaultdict()
            emotions_dict[face.get_id()] = defaultdict()
            measurements_dict[face.get_id()].update(face.get_measurements())
            expressions_dict[face.get_id()].update(face.get_expressions())
            emotions_dict[face.get_id()].update(face.get_emotions())
            mood_dict[face.get_id()] = Mood(mood=face.get_mood(), confidence=face.get_confidence(), dominant_emotion=face.get_dominant_emotion().dominant_emotion,
                                            dominant_emotion_confidence=face.get_dominant_emotion().confidence)
            bounding_box_dict[face.get_id()] = [face.get_bounding_box()[0].x,
                                                face.get_bounding_box()[0].y,
                                                face.get_bounding_box()[1].x,
                                                face.get_bounding_box()[1].y]

    def image_captured(self, image):
        global capture_last_ts
        capture_fps = 1000.0 / (image.timestamp() - capture_last_ts)
        time_metrics_dict['cfps'] = capture_fps
        capture_last_ts = image.timestamp()



def get_command_line_parameters(args):
    """
    read parameters entered on the command line.

        Parameters
        ----------
        args: argparse
            object of argparse module

        Returns
        -------
        tuple of str values
            details about input file name, data directory, num of faces to detect, output file name
    """
    if not args.video is None:
        input_file = args.video
        if not os.path.isfile(input_file):
            raise ValueError("Please provide a valid input video file")
    else:
        input_file = int(args.camera)
    data = args.data
    if not os.path.isdir(data):
        raise ValueError("Please check your data directory path")
    max_num_of_faces = int(args.num_faces)
    output_file = args.output
    csv_file = args.file
    frame_width = int(args.res[WIDTH])
    frame_height= int(args.res[HEIGHT])
    return input_file, data, max_num_of_faces, csv_file, output_file, frame_width, frame_height



def draw_bounding_box(frame):
    """
    For each frame, draw the bounding box on screen.

        Parameters
        ----------
        frame: affvisionPy.Frame
            Frame object to draw the bounding box on.

    """
    for fid, bb_points in bounding_box_dict.items():
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid)
        for key in emotions_dict[fid]:
            if 'valence' in str(key):
                valence_value = emotions_dict[fid][key]
            if 'anger' in str(key):
                anger_value = emotions_dict[fid][key]
            if 'joy' in str(key):
                joy_value = emotions_dict[fid][key]
        if valence_value < 0 and anger_value >= THRESHOLD_VALUE_FOR_EMOTIONS:
            cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (0, 0, 255), 3)
        elif valence_value >= THRESHOLD_VALUE_FOR_EMOTIONS and joy_value >= THRESHOLD_VALUE_FOR_EMOTIONS:
            cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (0, 255, 0), 3)
        else:
            cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (21, 169, 167), 3)



def get_bounding_box_points(fid):
    """
    Fetch upper_left_x, upper_left_y, upper_right_x, upper_right_y points of the bounding box.

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

    key_name = key.split(".")[1]
    key_text_width, key_text_height = get_text_size(key_name, cv2.FONT_HERSHEY_SIMPLEX, 1)
    val_text = str(round(val, 2))
    val_text_width, val_text_height = get_text_size(val_text, cv2.FONT_HERSHEY_SIMPLEX, 1)

    key_val_width = key_text_width + val_text_width

    cv2.putText(frame, key_name + ": ", (abs(x1 - key_val_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (255, 255, 255))
    cv2.putText(frame, val_text, (abs(x1 - val_text_width + PADDING_FOR_SEPARATOR), upper_left_y),
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

    for i in range(0, rounded_val + 1):
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
        for i in range(0, rounded_val):
            start_box_point_x += 10
            cv2.rectangle(overlay, (start_box_point_x, upper_right_y),
                          (start_box_point_x + width, upper_right_y - height), (186, 186, 186), -1)
            cv2.rectangle(overlay, (start_box_point_x, upper_right_y),
                          (start_box_point_x + width, upper_right_y - height), (0, 204, 102), -1)
        for i in range(rounded_val + 1, 10):
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



def write_metrics(frame):
    """
    write measurements, emotions, expressions on screen

        Parameters
        ----------
        frame: affvisionpy.Frame
            frame to write the metrics on

    """
    for fid in measurements_dict.keys():
        measurements = measurements_dict[fid]
        expressions = expressions_dict[fid]
        emotions = emotions_dict[fid]
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid)
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



def run(csv_data):
    """
    Starting point of the program, initializes the detector, processes a frame and then writes metrics to frame

        Parameters
        ----------
        csv_data: list
            Values to hold for each frame
    """
    args = parse_command_line()
    input_file, data, max_num_of_faces, csv_file, output_file, frame_width, frame_height = get_command_line_parameters(args)
    if isinstance(input_file, int):
        start_time = time.time()
    detector = af.SyncFrameDetector(data, max_num_of_faces)

    detector.enable_features({af.Feature.expressions, af.Feature.emotions})

    list = Listener()
    detector.set_image_listener(list)

    detector.start()

    captureFile = cv2.VideoCapture(input_file)
    window = cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processed Frame', frame_width, frame_height)

    if not args.video:
        captureFile.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        captureFile.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        #If cv2 silently fails, default to 1280 x 720 instead of 640 x 480
        if captureFile.get(3) != frame_width or captureFile.get(4) != frame_height:
            print(f"{frame_width} x {frame_height} is an unsupported resolution, defaulting to 1280 x 720")
            captureFile.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
            captureFile.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
            frame_width = DEFAULT_FRAME_WIDTH
            frame_height = DEFAULT_FRAME_HEIGHT
            cv2.resizeWindow('Processed Frame',DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT)
        file_width = frame_width
        file_height = frame_height

    else:
        file_width = int(captureFile.get(3))
        file_height = int(captureFile.get(4))
        #cv2.resizeWindow('Processed Frame', file_width,file_height)

    if output_file is not None:
       out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (file_width, file_height))
    count = 0

    while captureFile.isOpened():
        # Capture frame-by-frame
        ret, frame = captureFile.read()

        if ret == True:

            height = frame.shape[0]
            width = frame.shape[1]
            if isinstance(input_file, int):
                timestamp = (time.time() - start_time) * 1000.0
            else:
                timestamp = int(captureFile.get(cv2.CAP_PROP_POS_MSEC))
            time_metrics_dict['timestamp'] = timestamp #.put(timestamp)
            afframe = af.Frame(width, height, frame, af.ColorFormat.bgr, int(timestamp))
            count += 1
            try:
                detector.process(afframe)

            except Exception as exp:
                print(exp)
            write_metrics_to_csv_data_list(csv_data, round(timestamp, 0))

            if len(num_faces) > 0 and not check_bounding_box_outside(width, height):
                draw_bounding_box(frame)
                draw_affectiva_logo(frame, width, height)
                write_metrics(frame)
                cv2.imshow('Processed Frame', frame)
            else:
                draw_affectiva_logo(frame, width, height)
                cv2.imshow('Processed Frame', frame)
            if output_file is not None:
                out.write(frame)

            clear_all_dictionaries()

            if cv2.waitKey(1) == 27:
                break
        else:
            break

    captureFile.release()
    cv2.destroyAllWindows()
    detector.stop()

    # If video file is provided as an input

    if csv_file == "default":
        if os.sep in csv_file:
            csv_file = str(input_file.rsplit(os.sep, 1)[1])
        csv_file = csv_file.split(".")[0]
        write_csv_data_to_file(csv_data, csv_file)
    else:
        write_csv_data_to_file(csv_data, csv_file)



def clear_all_dictionaries():
    """
    Clears the dictionary values
    """
    bounding_box_dict.clear()
    emotions_dict.clear()
    expressions_dict.clear()
    measurements_dict.clear()
    mood_dict.clear()



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
    logo = cv2.imread("Final logo - RGB Magenta.png")
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



def check_bounding_box_outside(width, height):
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
        x1, y1, x2, y2 = get_bounding_box_points(fid)
        if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
            return True
        return False



def write_metrics_to_csv_data_list(csv_data, timestamp):
    """
    Write metrics per frame to a list

        Parameters
        ----------
        csv_data:
          list of per frame values to write to
        timestamp: int
           timestamp of each frame

    """
    global header_row
    current_frame_data = {}
    if not measurements_dict.keys():
        for field in header_row:
            if field == "TimeStamp":
                current_frame_data[field] = timestamp
            else:
                current_frame_data[field] = NOT_A_NUMBER
        csv_data.append(current_frame_data)
    else:
        for fid in measurements_dict.keys():
            current_frame_data["TimeStamp"] = timestamp
            current_frame_data["faceId"] = fid
            upperLeftX, upperLeftY, lowerRightX, lowerRightY = get_bounding_box_points(fid)
            current_frame_data["upperLeftX"] = upperLeftX
            current_frame_data["upperLeftY"] = upperLeftY
            current_frame_data["lowerRightX"] = lowerRightX
            current_frame_data["lowerRightY"] = lowerRightY
            for key,val in measurements_dict[fid].items():
                current_frame_data[str(key).split('.')[1]] = round(val,4)
            for key,val in emotions_dict[fid].items():
                current_frame_data[str(key).split('.')[1]] = round(val,4)
            for key,val in expressions_dict[fid].items():
                current_frame_data[str(key).split('.')[1]] = round(val,4)
            current_frame_data["mood"] = str(mood_dict[fid].mood).split('.')[1]
            current_frame_data["dominant_emotion_confidence"] = round(mood_dict[fid].dominant_emotion_confidence,4)
            current_frame_data["dominant_emotion"] = str(mood_dict[fid].dominant_emotion).split('.')[1]
            current_frame_data["confidence"] = round(mood_dict[fid].confidence,4)
            csv_data.append(current_frame_data)



def parse_command_line():
    """
    Make the options for command line

    Returns
    -------
    args: argparse object of the command line
    """
    parser = argparse.ArgumentParser(description="Sample code for demoing affdexface on webcam or a saved video file.\n \
        By default, the program will run with the camera parameter displaying frames of size 1280 x 720.\n \
        A CSV file will also be written by default with the filename 'default.csv'. ")
    parser.add_argument("-d", "--data", dest="data", required=True, help="path to directory containing the models")
    parser.add_argument("-i", "--input", dest="video", required=False,
                        help="path to input video file")
    parser.add_argument("-n", "--num_faces", dest="num_faces", required=False, default=1,
                        help="number of faces to identify in the frame")
    parser.add_argument("-c", "--camera", dest="camera", required=False, const="0", nargs='?', default=0,
                        help="enable this parameter take input from the webcam and provide a camera id for the webcam")
    parser.add_argument("-o", "--output", dest="output", required=False,
                        help="name of the output video file")
    parser.add_argument("-f", "--file", dest="file", required=False, default=DEFAULT_FILE_NAME,
                        help="name of the output CSV file")
    parser.add_argument("-r", "--resolution", dest='res', metavar=('width', 'height'), nargs=2, default=[1280, 720], help="resolution in pixels (2-values): width height")
    args = parser.parse_args()
    return args



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

if __name__ == "__main__":
    csv_data = list()
    run(csv_data)
