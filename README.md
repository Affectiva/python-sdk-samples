# Sample Python App for Affectiva Emotion SDK 5.0 #

This script demonstrates how to use Affectiva's affvisionpy module to process frames from either a webcam or a video file. It supports displaying the input frames on screen, overlaid with metric values, and can also write the metric values to an output CSV file as well as a video AVI file.

## Steps to install the app: ##

1. Make sure you have **affvisionpy**

2. **cd** into the directory having the requirements.txt file and run **pip3 install -r requirements.txt**

## Setting LD_LIBRARY_PATH
To use the "identity" or "appearances" features of the affvisionpy module, you must export the LD_LIBRARY_PATH
environment variable with a value that refers to the "lib" subfolder under the affvisionpy installation directory.

This installation directory can vary depending on your python configuration, but if you execute the following
command in a bash shell, it will set the environment value for you:

    export LD_LIBRARY_PATH=$(python3 -c "exec(\"import affvisionpy,os\nprint(os.path.dirname(os.path.realpath(affvisionpy.__file__)) + '/lib')\")")

## Usage ##

        python3 affvisionpy_sample.py <arguments>

run with -h or --help to see documentation for all supported arguments        
 
        python3 affvisionpy_sample.py -h

## Example Usages:

Run the script with a webcam.  **Note:** If the camera id is not supplied, by default the camera_id is set to 0.
    
    python3 affvisionpy_sample.py -d </path/to/data/dir> -c <camera_id> -n <num_of_faces_to_detect>

Run the script with a video file.

    python3 affvisionpy_sample.py -d </path/to/data/vision> -n <num_of_faces_to_detect> -i </path/to/video/file>

Run the script with the default webcam.

    python3 affvisionpy_sample.py -d </path/to/data/dir>
    
Run the script with the default webcam, limiting face detection to 1 face
    
    python3 affvisionpy_sample.py -d </path/to/data/dir> -n 1

Run the script with a 2nd webcam
    
    python3 affvisionpy_sample.py -d </path/to/data/dir> -c 1
        
Run the script with a webcam and save the metrics to a CSV file named metrics.csv

    python3 affvisionpy_sample.py -d </path/to/data/dir> -f metrics.csv

Run the script with a video file and save output video a file named metrics.avi

    python3 affvisionpy_sample.py -d </path/to/data/dir> -i myvideo.mp4 -o metrics.avi


## Additional Notes ##

For the video option, a CSV file is written with the metrics. By default, the CSV file will be named with the same name as the video file.

For both video and webcam options, the script displays real-time metrics on the screen.

For the video option, the synchronous version of affvisionpy's FrameDetector will be used (SyncFrameDetector). 
For the webcam option, the asynchronous version will be used (FrameDetector).
