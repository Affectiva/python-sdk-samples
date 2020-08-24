# !/usr/bin/env python3.5
import argparse
import sys
import os
import time
import csv

import affvisionpy as af
import cv2
 
from listener import Listener
from display_metrics import (draw_metrics, check_bounding_box_outside, draw_bounding_box, draw_affectiva_logo, get_bounding_box_points)
 

# Constants
NOT_A_NUMBER = 'nan'
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_FILE_NAME = "default"
DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR"
 
header_row = ['TimeStamp', 'faceId', 'upperLeftX', 'upperLeftY', 'lowerRightX', 'lowerRightY', 'confidence', 'interocular_distance',
        'pitch', 'yaw', 'roll', 'joy', 'anger', 'surprise', 'valence', 'fear', 'sadness', 'disgust', 'neutral', 'contempt', 'smile',
        'brow_raise', 'brow_furrow', 'nose_wrinkle', 'upper_lip_raise', 'mouth_open', 'eye_closure', 'cheek_raise', 'yawn',
        'blink', 'blink_rate', 'eye_widen', 'inner_brow_raise', 'lip_corner_depressor'
        ]
 
#Argparse Variable Constants
WIDTH = 0
HEIGHT = 1


def run(csv_data):
    """
    Starting point of the program, initializes the detector, processes a frame and then writes metrics to frame
 
        Parameters
        ----------
        csv_data: list
            Values to hold for each frame
    """
    parser, args = parse_command_line()
    input_file, data, max_num_of_faces, csv_file, output_file, frame_width, frame_height = get_command_line_parameters(parser, args)
    if isinstance(input_file, int):
        start_time = time.time()         
        detector = af.FrameDetector(data, max_num_faces=max_num_of_faces)
    else:
        detector = af.SyncFrameDetector(data, max_num_of_faces)

    detector.enable_features({af.Feature.expressions, af.Feature.emotions})
 
    listener = Listener()
    detector.set_image_listener(listener)
 
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
                listener.mutex.acquire()
                listener.time_metrics_dict['timestamp'] = timestamp 
                listener.mutex.release()
                    
                afframe = af.Frame(width, height, frame, af.ColorFormat.bgr, int(timestamp))
                count += 1
                  
                try:
                    detector.process(afframe)
 
                except Exception as exp:
                    print(exp)

                num_faces = listener.num_faces
                measurements_dict = listener.measurements_dict.copy()
                expressions_dict = listener.expressions_dict.copy()
                emotions_dict = listener.emotions_dict.copy()
                bounding_box_dict = listener.bounding_box_dict.copy()
                listener_metrics = {
                    "measurements": measurements_dict,
                    "expressions": expressions_dict,
                    "emotions": emotions_dict,
                    "bounding_box": bounding_box_dict
                }

                write_metrics_to_csv_data_list(csv_data, round(timestamp, 0), listener_metrics)

                if len(num_faces) > 0 and not check_bounding_box_outside(width, height, bounding_box_dict):
                    draw_bounding_box(frame, listener_metrics)
                    draw_affectiva_logo(frame, width, height)
                    draw_metrics(frame, listener_metrics)
                    cv2.imshow('Processed Frame', frame)
                else:
                    draw_affectiva_logo(frame, width, height)
                    cv2.imshow('Processed Frame', frame)
                if output_file is not None:
                    out.write(frame)
 
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
 

def parse_command_line():
    """
    Make the options for command line
 
    Returns
    -------
    args: argparse object of the command line
    """
    parser = argparse.ArgumentParser(description="Sample code for demoing affvisionpy module on webcam or a saved video file.\n \
        By default, the program will run with the camera parameter displaying frames of size 1280 x 720.\n")
    parser.add_argument("-d", "--data", dest="data", required=False, help="path to directory containing the models. \
                        Alternatively, specify the path via the environment variable " + DATA_DIR_ENV_VAR + "=/path/to/data/vision")
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
    return parser, args
 

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
    max_num_of_faces = int(args.num_faces)
    output_file = args.output
    csv_file = args.file
    frame_width = int(args.res[WIDTH])
    frame_height= int(args.res[HEIGHT])
    return input_file, data, max_num_of_faces, csv_file, output_file, frame_width, frame_height
 

if __name__ == "__main__":
    csv_data = list()
    run(csv_data)
