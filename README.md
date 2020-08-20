# Sample Python App for Affdex SDK #

This script demonstrates how to use Affectiva's affvisionpy module to process frames from either a webcam or a video file. It supports displaying the input frames on screen, overlaid with metric values, and can also write the metric values to an output CSV file as well as a video AVI file.

## Steps to install the app: ##

1. Make sure you have **affvisionpy**

2. **cd** into the directory having the requirements.txt file and run **pip3 install -r requirements.txt**


## Steps to run the script: ##

1. usage:

        python3 affvisionpy-sample.py [-h] -d DATA [-i VIDEO] [-n NUM_FACES] [-c [CAMERA]] [-o OUTPUT] [-f FILE] [-r WIDTH HEIGHT]

        required arguments:
            -d DATA, --data DATA  path to directory containing the models. Alternatively specify path via the environment variable AFFECTIVA_VISION_DATA_DIR=/path/to/data/vision

        optional arguments:

          -h, --help    show this help message and exit

          -i VIDEO, --input VIDEO    path to input video file


          -n NUM_FACES, --num_faces NUM_FACES    number of faces to identify in the frame


          -o OUTPUT, --output OUTPUT    enable this parameter to save the output video in a AVI file of your choice


          -f FILE, --file FILE    enable this parameter to save the output metrics in a CSV file of your choice


          -c [CAMERA], --camera [CAMERA]    enable this parameter to take input from the webcam and provide a camera id for the webcam


          -r --resolution WIDTH HEIGHT   set the resolution in pixels (width, height) for the webcam. Defaults to 1280 x 720 resolution.

          --sync    enable this flag to use SyncFrameDetector when using a webcam. Processing a video will automatically use SyncFrameDetector.

    **Note**: if only the data argument is supplied, the script defaults the run to a webcam and 1 face detection, displaying frames
    at default size of 1280 x 720. Only certain standard frame sizes are supported. For any unsupported frame sizes, the webcam frame
    size defaults to 1280 x 720. If any other configuration is required, it can be done using optional arguments.



2. We can use the same script to enable camera as well as input video.

3. By default the num of faces detected by the script is 1.

4. Example Usages:

    i. Command to run the script with webcam:

            python3 affvisionpy-sample.py -d </path/to/data/vision> -c <camera_id> -n <num_of_faces_to_detect>

       **Note:** If the camera id is not supplied, by default the camera_id is set to 0.

    ii. Command to run the script with a video file:

            python3 affvisionpy-sample.py -d </path/to/data/vision> -n <num_of_faces_to_detect> -i </path/to/video/file>


    iii. Command to run the script with a webcam and save the CSV file to a filename of your choice:

         python3 affvisionpy-sample.py -d </path/to/data/vision> -c <camera_id> -n <num_of_faces_to_detect> -f <filename>

    iv. Command to run the script with a video file and save output video AVI file to a filename of your choice:

         python3 affvisionpy-sample.py -d </path/to/data/vision> -n <num_of_faces_to_detect> -i </path/to/video/file> -o <filename>

    v. Command to run the script with a webcam and save the CSV file and output video AVI file to filenames of your choice:

        python3 affvisionpy-sample.py -d </path/to/data/vision> -n <num_of_faces_to_detect> -i </path/to/video/file> -o <filename> -f <filename>


## Additional Notes ##

5.  For the video option, a CSV file is written with the metrics. By default, the CSV file will be named with the same name as the video file.

6.  For both video and webcam options, the script displays real-time metrics on the screen.

7.  To get the output metrics in a file of your choice, give the name of the file with -f on the command line. If a CSV file is desired for the webcam option, the -f flag must be supplied.

8.  To get the output video, use the option -o along with the output file name.
