# !/usr/bin/env python3
from collections import defaultdict
from threading import Lock
import affvisionpy as af

BODY_POINTS = ['nose', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip',
               'left_hip', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'neck']

EDGES = [('neck', 'right_shoulder'), ('neck', 'left_shoulder'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
         ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('neck', 'right_hip'), ('neck', 'left_hip'),
         ('neck', 'nose'), ('nose', 'right_eye'), ('right_eye', 'right_ear'), ('nose', 'left_eye'), ('left_eye', 'left_ear')]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],  [0, 255, 0],
          [0, 255, 255], [0, 0, 255],  [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170]]

class BodyListener(af.BodyListener):
    """
    BodyListener class that return body metrics for processed frames.

    """

    def __init__(self, body_interval):
        super(BodyListener, self).__init__()

        self.count = 0
        self.process_last_ts = 0.0
        self.body_interval = body_interval
        self.mutex = Lock()
        self.bodyPoints = defaultdict()


    def results_updated(self, bodies, frame):
        timestamp = frame.timestamp()

        self.mutex.acquire()
        # avoid div by 0 error on the first frame
        try:
            process_fps = 1000.0 / (frame.timestamp() - self.process_last_ts)
        except:
            process_fps = 0
        print("timestamp:" + str(round(timestamp, 0)), "Frame " + str(self.count),
              "pfps: " + str(round(process_fps, 0)))

        self.count += 1
        self.process_last_ts = frame.timestamp()
        self.bodies = bodies
        self.clear_all_dictionaries()

        for body_id, body in bodies.items():
            b_pts = body.get_body_points()
            body_points = {}
            for b_pt, pt in b_pts.items():
                body_points[b_pt.name] = [int(pt.x), int(pt.y)]
            self.bodyPoints[body_id] = body_points


        self.mutex.release()

    def get_callback_interval(self):
        return self.body_interval

    def clear_all_dictionaries(self):
        """
        Clears the dictionary values
        """
        self.bodyPoints.clear()
