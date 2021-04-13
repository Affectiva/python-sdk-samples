# !/usr/bin/env python3
from collections import defaultdict
from threading import Lock
import affvisionpy as af

class FaceListener(af.FaceListener):
    """
    Listener class that return metrics for processed frames.
 
    """

    def __init__(self):
        super(FaceListener, self).__init__()

        self.count = 0
        self.process_last_ts = 0.0
        self.mutex = Lock()

        self.measurements_dict = defaultdict()
        self.expressions_dict = defaultdict()
        self.emotions_dict = defaultdict()
        self.bounding_box_dict = defaultdict()
        self.faces = defaultdict()

    def results_updated(self, faces, frame):
        timestamp = frame.timestamp()

        self.mutex.acquire()
        # avoid div by 0 error on the first frame
        try:
            process_fps = 1000.0 / (frame.timestamp() - self.process_last_ts)
        except:
            process_fps = 0
        print("timestamp:" + str(round(timestamp, 0)), "Frame " + str(self.count), "pfps: " + str(round(process_fps, 0)))
        self.count += 1
        self.process_last_ts = frame.timestamp()
        self.faces = faces

        self.clear_all_dictionaries()
        for fid, face in faces.items():
            self.measurements_dict[fid] = defaultdict()
            self.expressions_dict[fid] = defaultdict()
            self.emotions_dict[fid] = defaultdict()

            self.measurements_dict[fid].update(face.get_measurements())
            self.expressions_dict[fid].update(face.get_expressions())
            self.emotions_dict[fid].update(face.get_emotions())
            self.bounding_box_dict[fid] = [face.get_bounding_box()[0].x,
                                           face.get_bounding_box()[0].y,
                                           face.get_bounding_box()[1].x,
                                           face.get_bounding_box()[1].y,
                                           face.get_confidence()]

        self.mutex.release()

    def face_lost(self, timestamp, face_id):
        pass

    def face_found(self, timestamp, face_id):
        pass

    def clear_all_dictionaries(self):
        """
        Clears the dictionary values
        """
        self.measurements_dict.clear()
        self.expressions_dict.clear()
        self.emotions_dict.clear()
        self.bounding_box_dict.clear()
