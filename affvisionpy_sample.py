# !/usr/bin/env python3

import argparse
import sys
import os
import time
import csv
import affvisionpy as af
import cv2
import math

from face_listener import FaceListener as ImageListener

from display_metrics import (draw_metrics, check_bounding_box_outside, draw_bounding_box, draw_affectiva_logo,
                             get_affectiva_logo, get_bounding_box_points)

# Constants
NOT_A_NUMBER = 'nan'
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080
DEFAULT_FILE_NAME = "default"
DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR"

HEADER_ROW_FACES = ['TimeStamp', 'faceId', 'upperLeftX', 'upperLeftY', 'lowerRightX', 'lowerRightY', 'confidence',
                    'interocular_distance', 'pitch', 'yaw', 'roll', 'joy', 'anger', 'surprise', 'valence', 'fear',
                    'sadness', 'disgust', 'neutral', 'contempt', 'smile', 'brow_raise', 'brow_furrow', 'nose_wrinkle',
                    'upper_lip_raise', 'mouth_open', 'eye_closure', 'cheek_raise', 'lid_tighten', 'blink', 'blink_rate',
                    'eye_widen', 'inner_brow_raise', 'lip_corner_depressor', 'attention', 'jaw_drop', 'smirk', 'chain_raise',
                    'engagement', 'lip_pucker', 'dimpler', 'lip_stretch', 'lip_press', 'lip_suck'
                    ]

header_row = []

def get_video_fps(input_file, fps):
    """
    get fps related to input file

        Parameters
        ----------
        input_file: input file name to extract fps value

        fps: frames per second value

    """

    # Start default camera
    video = cv2.VideoCapture(input_file)

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Release video
    video.release()

def run(csv_data):
    """
    Starting point of the program, initializes the detector, processes a frame and then writes metrics to frame

        Parameters
        ----------
        csv_data: list
            Values to hold for each frame
    """
    parser, args = parse_command_line()
    input_file, data_dir, max_num_of_faces, csv_file, output_file, frame_width, frame_height = get_command_line_parameters(
        parser, args)

    start_time = 0

    if isinstance(input_file, int):
        start_time = time.time()
        baseline_mode = args.baseline_mode if args.baseline_mode is not None else True
        detector = af.FrameDetector(data_dir, 
                                    af.BaselineMode.ON if baseline_mode else af.BaselineMode.OFF, 
                                    max_num_faces=max_num_of_faces)
    else:
        baseline_mode = args.baseline_mode if args.baseline_mode is not None else is_input_file_a_video(input_file)
        detector = af.SyncFrameDetector(data_dir, 
                                        af.BaselineMode.ON if baseline_mode else af.BaselineMode.OFF, 
                                        max_num_faces=max_num_of_faces)

    fps = 30
    if args.video:
        get_video_fps(input_file, fps)
    capture_file = cv2.VideoCapture(input_file)

    if not args.video:
        capture_file.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        capture_file.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        # If cv2 silently fails, default to 1920 x 1080 instead of 640 x 480
        if capture_file.get(3) != frame_width or capture_file.get(4) != frame_height:
            print(capture_file.get(3), "x", capture_file.get(4), "is an unsupported resolution, defaulting to 1920 x 1080")
            capture_file.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
            capture_file.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
            frame_width = DEFAULT_FRAME_WIDTH
            frame_height = DEFAULT_FRAME_HEIGHT

        file_width = frame_width
        file_height = frame_height

    else:
        file_width = int(capture_file.get(3))
        file_height = int(capture_file.get(4))

    out = None
    if output_file is not None:
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (file_width, file_height))

    logo = get_affectiva_logo(file_width, file_height)

    process_face_input(detector, capture_file, input_file, start_time, output_file, out, logo, args)

    capture_file.release()
    cv2.destroyAllWindows()

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

def is_input_file_a_video(input_file):
    capture_file = cv2.VideoCapture(input_file)
    if capture_file.isOpened():
        # if we successfully read two frames from this file, we can assume it is a video (as opposed to an image)
        _, _ = capture_file.read()
        ret, frame = capture_file.read()
        return ret
    else:
        return False

def process_face_input(detector, capture_file, input_file, start_time, output_file, out, logo, args):
    count = 0
    last_timestamp = 0

    features = {af.Feature.expressions, af.Feature.emotions}
    detector.enable_features(features)

    listener = ImageListener()
    detector.set_image_listener(listener)

    detector.start()

    while capture_file.isOpened():
        # Capture frame-by-frame
        ret, frame = capture_file.read()

        if ret == True:

            height = frame.shape[0]
            width = frame.shape[1]
            # if the input is a webcam, or a photo
            if isinstance(input_file, int) or not is_input_file_a_video(input_file):
                curr_timestamp = (time.time() - start_time) * 1000.0
            else:
                curr_timestamp = int(capture_file.get(cv2.CAP_PROP_POS_MSEC))
            if curr_timestamp > last_timestamp or count == 0:  # if there's a problem with the timestamp, don't process the frame

                last_timestamp = curr_timestamp
                afframe = af.Frame(int(curr_timestamp), width, height, frame, af.Frame.ColorFormat.bgr, af.Frame.Rotation.upright)
                count += 1

                try:
                    detector.process(afframe)

                except Exception as exp:
                    print(exp)

                listener.mutex.acquire()

                faces = listener.faces.copy()
                measurements_dict = listener.measurements_dict.copy()
                expressions_dict = listener.expressions_dict.copy()
                emotions_dict = listener.emotions_dict.copy()
                bounding_box_dict = listener.bounding_box_dict.copy()

                listener.mutex.release()

                listener_metrics = {
                    "measurements": measurements_dict,
                    "expressions": expressions_dict,
                    "emotions": emotions_dict,
                    "bounding_box": bounding_box_dict
                }

                write_face_metrics_to_csv_data_list(csv_data, round(curr_timestamp, 0), listener_metrics)

                draw_affectiva_logo(frame, logo, width, height)
                if len(faces) > 0 and not check_bounding_box_outside(width, height, bounding_box_dict):
                    draw_bounding_box(frame, listener_metrics)
                    draw_metrics(frame, listener_metrics)

                if not args.no_draw:
                    cv2.imshow('Processed Frame', frame)

                if output_file is not None:
                    out.write(frame)

                # wait for user input only if the input file is a photo
                waitkey_delay = 1 if isinstance(input_file, int) or is_input_file_a_video(input_file) else 0
                if cv2.waitKey(waitkey_delay) == 27:
                    break
            else:
                print("skipped a frame due to the timestamp not incrementing - old timestamp %f, current timestamp %f" %
                      (last_timestamp, curr_timestamp))
        else:
            break

    detector.stop()

def write_face_metrics_to_csv_data_list(csv_data, timestamp, listener_metrics):
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
    current_frame_data = {}
    if not listener_metrics["measurements"].keys():
        write_default_csv_data(current_frame_data, timestamp)
    else:
        for fid in listener_metrics["measurements"].keys():
            current_frame_data["TimeStamp"] = timestamp
            current_frame_data["faceId"] = fid
            write_bbox_metrics_to_csv(fid, listener_metrics, current_frame_data)

            for key, val in listener_metrics["measurements"][fid].items():
                current_frame_data[key.name] = round(val, 4)
            for key, val in listener_metrics["emotions"][fid].items():
                current_frame_data[key.name] = round(val, 4)
            for key, val in listener_metrics["expressions"][fid].items():
                current_frame_data[key.name] = round(val, 4)
            current_frame_data["confidence"] = round(listener_metrics["bounding_box"][fid][4], 4)

            csv_data.append(current_frame_data)
            current_frame_data = {}

def write_bbox_metrics_to_csv(id, listener_metrics, current_frame_data):
    upperLeftX, upperLeftY, lowerRightX, lowerRightY = get_bounding_box_points(id,
                                                                               listener_metrics["bounding_box"])
    current_frame_data["upperLeftX"] = upperLeftX
    current_frame_data["upperLeftY"] = upperLeftY
    current_frame_data["lowerRightX"] = lowerRightX
    current_frame_data["lowerRightY"] = lowerRightY

def write_default_csv_data(current_frame_data, timestamp):
    current_frame_data["TimeStamp"] = timestamp
    for field in header_row[1:]:
        current_frame_data[field] = NOT_A_NUMBER
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
        # When writing a row, restval is the default value when it isn't in the dict row.
        # extrasaction='ignore' prevents complaining if all columns are not present in dict row
        writer = csv.DictWriter(c_file, fieldnames=header_row, restval='o', extrasaction='ignore')
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    c_file.close()

def parse_command_line():
    """
    Make the options for command line

    Returns
    -------
    args: argparse object of the command line
    """
    parser = argparse.ArgumentParser(description="Sample code for demoing affvisionpy module on webcam or a saved video file.\n \
        By default, the program will run with the camera parameter displaying frames of size 1920 x 1080.\n")
    parser.add_argument("-d", "--data", dest="data_dir", required=False,
                        help="path to SDK data directory. \
                        Defaults to the data directory packaged with the affvisionpy module. \
                        Alternatively, specify the path via the environment variable " + DATA_DIR_ENV_VAR + "=/path/to/data/")
    parser.add_argument("-i", "--input", dest="video", required=False,
                        help="path to input video file")
    parser.add_argument("-n", "--num_faces", dest="num_faces", required=False, default=5,
                        help="number of faces to identify in the frame")
    parser.add_argument("-c", "--camera", dest="camera", required=False, const="0", nargs='?', default=0,
                        help="enable this parameter take input from the webcam and provide a camera id for the webcam")
    parser.add_argument("-o", "--output", dest="output", required=False,
                        help="name of the output video file")
    parser.add_argument("-f", "--file", dest="file", required=False, default=DEFAULT_FILE_NAME,
                        help="name of the output CSV file")
    parser.add_argument("-r", "--resolution", dest='res', metavar=('width', 'height'), nargs=2, default=[1920, 1080],
                        help="resolution in pixels (2-values): width height")
    parser.add_argument("--no-draw", dest="no_draw", action='store_true', help="Don't draw window while processing video, default is set to False")
    parser.add_argument("--baseline_mode", dest="baseline_mode", required=False, type=lambda x: x.lower() == 'true')
    args = parser.parse_args()
    return parser, args

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
    if args.video is not None:
        input_file = args.video
        if not os.path.isfile(input_file):
            raise ValueError("Please provide a valid input video file")
    else:
        if str(args.camera).isdigit():
            input_file = int(args.camera)
        else:
            raise ValueError("Please provide an integer value for camera")

    # if a data dir was specified on the command line, use that.  If not, use the value specified via the env var.
    # if the env var wasn't specified either, use the "data" subfolder under the install location of the module.
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.environ.get(DATA_DIR_ENV_VAR)
        if data_dir is not None:
            print("Using data dir=\"" + data_dir + "\" from env var", DATA_DIR_ENV_VAR)
        else:
            # default to the data dir packaged with the affvisionpy module
            data_dir = os.path.dirname(af.__file__) + "/data"
    if not os.path.isdir(data_dir):
        print("ERROR: data directory \"" + data_dir + "\" does not exist\n")
        parser.print_help()
        sys.exit(1)

    global header_row
    header_row = HEADER_ROW_FACES

    max_num_of_faces = int(args.num_faces)
    output_file = args.output
    csv_file = args.file
    frame_width = int(args.res[0])
    frame_height = int(args.res[1])
    return input_file, data_dir, max_num_of_faces, csv_file, output_file, frame_width, frame_height

if __name__ == "__main__":
    csv_data = list()
    run(csv_data)
