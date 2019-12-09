# !/usr/bin/env python3.5
try:
    import TIS
except ImportError:
    print("TIS module is not imported")

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
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080
DEFAULT_FILE_NAME = "default"
DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR"
#used in tis_cam
FRAMERATE = 30

#Argparse Variable Constants
WIDTH = 0
HEIGHT = 1

process_last_ts = 0.0
capture_last_ts = 0.0



header_row = ['TimeStamp', 'faceId', 'identity', 'identity_confidence', 'age', 'age_confidence', 'age_category']
        

identity_dict = defaultdict()
age_dict = defaultdict()
age_category_dict = defaultdict()
bounding_box_dict = defaultdict()
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
            print("fid:" + str(fid))
            print("identity: " + str(face.get_identity().identity))
            print("age: " + str(face.get_age().age_metric))
            print(face.get_bounding_box()[0])

            identity_dict[face.get_id()] = face.get_identity()
            age_dict[face.get_id()] = face.get_age()
            age_category_dict[face.get_id()] =face.get_age_category()

            bounding_box_dict[face.get_id()] = [face.get_bounding_box()[0].x,
                                                face.get_bounding_box()[0].y,
                                                face.get_bounding_box()[1].x,
                                                face.get_bounding_box()[1].y,
                                                face.get_confidence()]

    def image_captured(self, image):
        global capture_last_ts
        capture_fps = 1000.0 / (image.timestamp() - capture_last_ts)
        time_metrics_dict['cfps'] = capture_fps
        capture_last_ts = image.timestamp()



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
    fisheye_serial = None
    input_file = None
    if args.video is not None:
        input_file = args.video
        if not os.path.isfile(input_file):
            raise ValueError("Please provide a valid input video file")
    elif args.fisheye_serial is None:
        input_file = int(args.camera)
    else:
        fishey_serial = args.fisheye_serial
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
    return input_file, fisheye_serial, data, max_num_of_faces, csv_file, output_file, frame_width, frame_height



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
        cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (21, 169, 167), 3)



def get_bounding_box_points(fid):
    """
    Fetch upper_left_x, upper_left_y, lower_right_x, lower_right_y points of the bounding box.

        Parameters
        ----------
        fid: int
            face id of the face to get the bounding box for

        Returns
        -------
        tuple of int values
            tuple with upper_left_x, upper_left_y, lower_right_x, lower_right_y values
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



def draw_metric_with_bar(destination_img, x, y, key, val):
    x += 50
    cv2.putText(destination_img, key, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.25, (0,0,0), thickness=5)
    cv2.putText(destination_img, key, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.25, (255,255,255), thickness=2)
    # negative values signify rectangle should be filled
    text_width = cv2.getTextSize(key,cv2.FONT_HERSHEY_PLAIN, 1.25, thickness=5)[0][0]
    x += text_width + 10
    for j in range(0,5):
        if (abs(val) > (j*10 + 50)):
            x_val = x + (j * 15)
            c = (0,255,0)
            cv2.rectangle(destination_img, (x_val, y - 10), (x_val + 10, y), c, -1)

    


def draw_metric_with_text(destination_img, x, y, key, val):
    descriptor = key + ': ' + str(val)
    cv2.putText(destination_img, descriptor, (x + 50, y + 5), cv2.FONT_HERSHEY_PLAIN, 1.25, (0,0,0), thickness=5)
    cv2.putText(destination_img, descriptor, (x + 50, y + 5), cv2.FONT_HERSHEY_PLAIN, 1.25, (255,255,255), thickness=2)

def visualize_identity(id):
	if id == -1:
		return "UNKNOWN"
	else:
		return str(id)

def visualize_age(age):
	if age == -1:
		return "UNKNOWN"
	else:
		return str(age)



def visualize_adult_bucket(age):
    if age >= 18 and age <36:
        return "Adult(18-36)"
    elif age >= 36 and age < 55:
        return "Adult(36-55)"
    elif age >= 55 and age < 76:
        return "Adult(55-76)"
    else: 
        return "Adult(76+)"

def visualize_age_category(age_category, age):
    print(type(age_category))
    if age_category == af.AgeCategory.unknown:
        return "UNKNOWN"
    elif age_category == af.AgeCategory.baby:
        return "Baby"
    elif age_category == af.AgeCategory.child:
        return "Child"
    elif age_category == af.AgeCategory.teen:
        return "Teen"
    else:
        return visualize_adult_bucket(age)




def write_metrics(frame):
    """
    write id, id_confidence, age, age_confidence, age_category on screen

        Parameters
        ----------
        frame: affvisionpy.Frame
            frame to write the metrics on

    """
    for fid in identity_dict.keys():
        identity_metric = identity_dict[fid]
        age_metric = age_dict[fid]
        age_category = age_category_dict[fid]
        
        left_x, upper_y, right_x, lower_y = get_bounding_box_points(fid)

        # draw identity
        draw_metric_with_text(frame, right_x, upper_y, "identity", visualize_identity(identity_metric.identity))
        upper_y += 25
        draw_metric_with_bar(frame, right_x, upper_y, "identity_confidence", identity_metric.confidence)
        upper_y += 25

        #draw age
        draw_metric_with_text(frame, right_x, upper_y, "age", visualize_age(age_metric.age_metric))
        upper_y += 25
        draw_metric_with_bar(frame, right_x, upper_y, "age_confidence", age_metric.confidence)
        upper_y += 25

        #draw age_category
        draw_metric_with_text(frame, right_x, upper_y, "age_category", visualize_age_category(age_category, age_metric.age_metric))



def run(csv_data):
    """
    Starting point of the program, initializes the detector, processes a frame and then writes metrics to frame

        Parameters
        ----------
        csv_data: list
            Values to hold for each frame
    """
    parser, args = parse_command_line()
    input_file, fisheye_serial, data, max_num_of_faces, csv_file, output_file, frame_width, frame_height = get_command_line_parameters(parser, args)
    #if isinstance(input_file, int) or fisheye_serial is not None:
    start_time = time.time()
    detector = af.SyncFrameDetector(data, max_num_of_faces)

    detector.enable_features({af.Feature.identity, af.Feature.appearances})

    listener = Listener()
    detector.set_image_listener(listener)

    detector.start()
    count = 0
    if args.fisheye_serial:
        Tis = TIS.TIS(args.fisheye_serial, frame_width, frame_height, FRAMERATE, True)
        Tis.Start_pipeline()  
        print('Press q to stop')


        while True:
        # Capture frame-by-frame
            if Tis.Snap_image(1/30) is True:
                frame = Tis.Get_image()  # Get the image. It is a numpy array
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                height = frame.shape[0]
                width = frame.shape[1]               
                timestamp = (time.time() - start_time) * 1000.0
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
            
        Tis.Stop_pipeline()
    else:
        captureFile = cv2.VideoCapture(input_file)
        window = cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
        
        if not args.video:
            cv2.resizeWindow('Processed Frame', frame_width, frame_height)
            captureFile.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            captureFile.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            #If cv2 silently fails, default to 1920 x 1080 instead of 640 x 480
            if captureFile.get(3) != frame_width or captureFile.get(4) != frame_height:
                print(captureFile.get(3), "x", captureFile.get(4), "is an unsupported resolution, defaulting to 1920 x 1080")
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
    identity_dict.clear()
    age_dict.clear()
    age_category_dict.clear()



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
    if not identity_dict.keys():
        current_frame_data = {}
        current_frame_data["TimeStamp"] = timestamp
        for field in header_row[1:]:
            current_frame_data[field] = NOT_A_NUMBER
        csv_data.append(current_frame_data)
    else:
        for fid in identity_dict:
            current_frame_data = {}
            current_frame_data["TimeStamp"] = timestamp
            current_frame_data["faceId"] = fid
            #leftX, upperY, rightX, lowerY = get_bounding_box_points(fid)
            #current_frame_data["leftX"] = leftX
            #current_frame_data["upperY"] = upperY
            #current_frame_data["rightX"] = rightX
            #current_frame_data["lowerY"] = lowerY

            current_frame_data["identity"] = identity_dict[fid].identity
            current_frame_data["identity_confidence"] = identity_dict[fid].confidence
            current_frame_data["age"] = age_dict[fid].age_metric
            current_frame_data["age_confidence"] = age_dict[fid].confidence
            current_frame_data["age_category"] = age_category_dict[fid]

            csv_data.append(current_frame_data)



def parse_command_line():
    """
    Make the options for command line

    Returns
    -------
    args: argparse object of the command line
    """
    parser = argparse.ArgumentParser(description="Sample code for demoing affvisionpy module on webcam or a saved video file.\n \
        By default, the program will run with the camera parameter displaying frames of size 1920 x 1080.\n")
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
    parser.add_argument("-r", "--resolution", dest='res', metavar=('width', 'height'), nargs=2, default=[1920, 1080], help="resolution in pixels (2-values): width height")
    parser.add_argument("-s", "--fisheyecam", dest='fisheye_serial', action="store", help="set as fisheye camera serial number if a fisheye camera is used" )
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
