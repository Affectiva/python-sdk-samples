# !/usr/bin/env python3.5
import argparse
import csv
import sys
import os
import time
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
DECIMAL_ROUNDING_FACTOR = 2
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_FILE_NAME = "default"
DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR"

#Argparse Variable Constants
WIDTH = 0
HEIGHT = 1

process_last_ts = 0.0
capture_last_ts = 0.0

header_row = ['TimeStamp', 'faceId', 'upperLeftX', 'upperLeftY', 'lowerRightX', 'lowerRightY', 'confidence',
              'interocular_distance',
              'pitch', 'yaw', 'roll', 'joy', 'anger', 'surprise', 'valence', 'fear', 'sadness', 'disgust', 'contempt',
              'neutral', 'smile',
              'brow_raise', 'brow_furrow', 'nose_wrinkle', 'upper_lip_raise', 'mouth_open', 'eye_closure',
              'cheek_raise', 'yawn',
              'blink', 'blink_rate', 'eye_widen', 'inner_brow_raise', 'lip_corner_depressor', 'gaze_region',
              'gaze_confidence', 'glasses'
              ]

measurements_dict = defaultdict()
expressions_dict = defaultdict()
emotions_dict = defaultdict()
bounding_box_dict = defaultdict()
time_metrics_dict = defaultdict()
num_faces = defaultdict()
identities_dict = defaultdict()
identity_names_dict = defaultdict()
gaze_metric_dict = defaultdict()
glasses_dict = defaultdict()


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
        #avoid div by 0 error on the first frame
        try:
            process_fps = 1000.0 / (image.timestamp() - process_last_ts)
        except:
            process_fps = 0
        print("timestamp:" + str(round(timestamp, 0)), "Frame " + str(count), "cfps: " + str(round(capture_fps, 0)), "pfps: " + str(round(process_fps, 0)))
        count +=1
        process_last_ts = image.timestamp()
        self.faces = faces
        global num_faces
        num_faces = faces
        for fid, face in faces.items():
            measurements_dict[fid] = defaultdict()
            expressions_dict[fid] = defaultdict()
            emotions_dict[fid] = defaultdict()
            measurements_dict[fid].update(face.get_measurements())
            expressions_dict[fid].update(face.get_expressions())
            emotions_dict[fid].update(face.get_emotions())
            bounding_box_dict[fid] = [face.get_bounding_box()[0].x,
                                      face.get_bounding_box()[0].y,
                                      face.get_bounding_box()[1].x,
                                      face.get_bounding_box()[1].y,
                                      face.get_confidence()]
            identities_dict[fid] = face.get_identity().identity
            gaze_metric_dict[fid] = face.get_gaze()
            glasses_dict[fid] = face.get_glasses()

    def image_captured(self, image):
        global capture_last_ts
        try:
            capture_fps = 1000.0 / (image.timestamp() - capture_last_ts)
        except:
            capture_fps = 0
        time_metrics_dict['cfps'] = capture_fps
        capture_last_ts = image.timestamp()


def read_identities_csv(data_dir):
    """Read the identities.csv file and return its contents (minus the header row) as a dict

    Parameters
    ----------
    data_dir: data directory path
    """
    lines = {}
    csv_path = data_dir + '/attribs/identities.csv'
    if os.path.isfile(csv_path):
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # skip header row
            for row in reader:
                lines[row[0]] = row[1]
    return lines

def get_command_line_parameters(parser, args):
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
    if not data:
        data = os.environ.get(DATA_DIR_ENV_VAR)
        if data == None:
            print("ERROR: Data directory not specified via command line or env var:", DATA_DIR_ENV_VAR, "\n")
            parser.print_help()
            sys.exit(1)
        print("Using value", data, "from env var", DATA_DIR_ENV_VAR)
    if not os.path.isdir(data):
        print("ERROR: Please check your data directory path\n")
        parser.print_help()
        sys.exit(1)

    if args.show_identity:
        global identity_names_dict
        # read in the csv file that maps identities to names
        identity_names_dict = read_identities_csv(data)
        global header_row
        header_row.extend(['identity', 'name'])

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
    for fid in bounding_box_dict.keys():
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

def display_identity_on_screen(frame, identity, upper_left_y, upper_left_x):
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


def write_metrics(args, frame):
    """
    write metrics on screen
 
        Parameters
        ----------
        args: parsed command line arguments
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

        if args.show_identity:
            display_identity_on_screen(frame, identities_dict[fid], upper_left_y, upper_left_x)

        for key, val in measurements.items():
            display_measurements_on_screen(key, val, upper_left_y, frame, upper_left_x)
            upper_left_y += 25

        for key, val in emotions.items():
            display_emotions_on_screen(key, val, upper_left_y, frame, upper_left_x)
            upper_left_y += 25

        gaze_reg = "gaze_region: " + str(gaze_metric_dict[fid].gaze_region.name)
        key_text_width, key_text_height = get_text_size(gaze_reg, cv2.FONT_HERSHEY_SIMPLEX, 1)

        cv2.putText(frame, gaze_reg , (abs(upper_left_x - key_text_width - PADDING_FOR_SEPARATOR), upper_left_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    TEXT_SIZE,
                    (255, 255, 255), 2, cv2.LINE_AA)
        upper_left_y += 25
        display_emotions_on_screen("g.gaze_confidence", gaze_metric_dict[fid].confidence, upper_left_y, frame, upper_left_x)
        upper_left_y += 25

        display_emotions_on_screen("g.glasses", glasses_dict[fid], upper_left_y, frame, upper_left_x)
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
    global num_faces
    parser, args = parse_command_line()
    input_file, data, max_num_of_faces, csv_file, output_file, frame_width, frame_height = get_command_line_parameters(parser, args)

    if isinstance(input_file, int):
        start_time = time.time()
    detector = af.SyncFrameDetector(data, max_num_of_faces)

    features = {af.Feature.expressions, af.Feature.emotions}
    if args.show_identity:
        features.add(af.Feature.identity)
    detector.enable_features(features)

    list = Listener()
    detector.set_image_listener(list)

    detector.start()

    captureFile = cv2.VideoCapture(input_file)
    window = cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)

    if not args.video:
        cv2.resizeWindow('Processed Frame', frame_width, frame_height)
        captureFile.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        captureFile.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        #If cv2 silently fails, default to 1280 x 720 instead of 640 x 480
        if captureFile.get(3) != frame_width or captureFile.get(4) != frame_height:
            print(frame_width, "x", frame_height, "is an unsupported resolution, defaulting to 1280 x 720")
            cv2.resizeWindow('Processed Frame',DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT)
            captureFile.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
            captureFile.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
            frame_width = DEFAULT_FRAME_WIDTH
            frame_height = DEFAULT_FRAME_HEIGHT

        file_width = frame_width
        file_height = frame_height

    else:
        file_width = int(captureFile.get(3))
        file_height = int(captureFile.get(4))
        cv2.resizeWindow('Processed Frame', file_width, file_height)

    if output_file is not None:
       out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (file_width, file_height))
    count = 0
    timestamp = 0
    last_timestamp = 0

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
            if timestamp>last_timestamp or count == 0: # if there's a problem with the timestamp, don't process the frame

                last_timestamp = timestamp
                time_metrics_dict['timestamp'] = timestamp #.put(timestamp)
                afframe = af.Frame(width, height, frame, af.ColorFormat.bgr, int(timestamp))
                count += 1

                try:
                    detector.process(afframe)

                except Exception as exp:
                    print(exp)
                write_metrics_to_csv_data_list(args, csv_data, round(timestamp, 0))

                if len(num_faces) > 0 and not check_bounding_box_outside(width, height):
                    draw_bounding_box(frame)
                    draw_affectiva_logo(frame, width, height)
                    write_metrics(args, frame)
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
                print("skipped a frame due to the timestamp not incrementing - old timestamp %f, new timestamp %f" % (last_timestamp,timestamp))
        else:
            break

    captureFile.release()
    cv2.destroyAllWindows()
    detector.stop()

    # If video file is provided as an input
    if not isinstance(input_file, int):
        if csv_file == DEFAULT_FILE_NAME:
            if os.sep in input_file:
                csv_file = str(input_file.rsplit(os.sep, 1)[1])
            csv_file = csv_file.split(".")[0]
        write_csv_data_to_file(csv_data, csv_file)
    else:
        if not csv_file == DEFAULT_FILE_NAME:
            write_csv_data_to_file(csv_data, csv_file)



def clear_all_dictionaries():
    """
    Clears the dictionary values
    """
    bounding_box_dict.clear()
    emotions_dict.clear()
    expressions_dict.clear()
    measurements_dict.clear()
    identities_dict.clear()
    gaze_metric_dict.clear()
    glasses_dict.clear()


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
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid)
        if upper_left_x < 0 or lower_right_x > width or upper_left_y < 0 or lower_right_y > height:
            return True
        return False



def write_metrics_to_csv_data_list(args, csv_data, timestamp):
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
    if not measurements_dict.keys():
        current_frame_data = {}
        current_frame_data["TimeStamp"] = timestamp
        for field in header_row[1:]:
            current_frame_data[field] = NOT_A_NUMBER
        csv_data.append(current_frame_data)
    else:
        for fid in measurements_dict.keys():
            current_frame_data = {}
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
            if args.show_identity:
                identity = identities_dict[fid]
                current_frame_data["identity"] = identity
                if str(identity) in identity_names_dict:
                    current_frame_data["name"] = identity_names_dict[str(identity)]
                else:
                    current_frame_data["name"] = "Unknown"
            current_frame_data["confidence"] = round(bounding_box_dict[fid][4], 4)
            current_frame_data["gaze_region"] = str(gaze_metric_dict[fid].gaze_region.name)
            current_frame_data["gaze_confidence"] = str(gaze_metric_dict[fid].confidence)
            current_frame_data["glasses"] = round(glasses_dict[fid])
            csv_data.append(current_frame_data)



def parse_command_line():
    """
    Make the options for command line
 
    Returns
    -------
    args: argparse object of the command line
    """
    parser = argparse.ArgumentParser(description="Sample code for demoing affvisionpy module using a webcam or a video file.\n \
        By default, the program will run with the camera parameter, displaying frames of size 1280 x 720.\n")
    parser.add_argument("-d", "--data", dest="data", required=False, help="path to SDK data directory \
                        Alternatively, specify the path via the environment variable " + DATA_DIR_ENV_VAR)
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
    parser.add_argument("--identity", dest="show_identity", action='store_true', help="show identity metrics")
    args = parser.parse_args()
    return parser, args



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
