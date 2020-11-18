# !/usr/bin/env python3
from collections import defaultdict
from threading import Lock
import affvisionpy as af

class FaceListener(af.ImageListener):
    """
    Listener class that return metrics for processed frames.
 
    """
    def __init__(self):
        super(FaceListener, self).__init__()
        
        self.count = 0
        self.process_last_ts = 0.0
        self.capture_last_ts = 0.0
        self.mutex = Lock()

        self.measurements_dict = defaultdict()
        self.expressions_dict = defaultdict()
        self.emotions_dict = defaultdict()
        self.bounding_box_dict = defaultdict()
        self.time_metrics_dict = defaultdict()
        self.faces = defaultdict()

        self.drowsiness_dict = defaultdict()
        self.identities_dict = defaultdict()
        self.gaze_metric_dict = defaultdict()
        self.glasses_dict = defaultdict()
        self.age_metric_dict = defaultdict()
        self.age_category_dict = defaultdict()
 
    def results_updated(self, faces, image):
        timestamp = image.timestamp()

        self.mutex.acquire()
        capture_fps = self.time_metrics_dict['cfps']
        #avoid div by 0 error on the first frame
        try:
            process_fps = 1000.0 / (image.timestamp() - self.process_last_ts)
        except:
            process_fps = 0
        print("timestamp:" + str(round(timestamp, 0)), "Frame " + str(self.count), "cfps: " + str(round(capture_fps, 0)), 
              "pfps: " + str(round(process_fps, 0)))
        self.count +=1
        self.process_last_ts = image.timestamp()
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
            self.drowsiness_dict[fid] = face.get_drowsiness()
            self.identities_dict[fid] = face.get_identity().identity
            self.gaze_metric_dict[fid] = face.get_gaze()
            self.glasses_dict[fid] = face.get_glasses()
            self.age_metric_dict[fid] = face.get_age()
            self.age_category_dict[fid] = face.get_age_category().name
            
        self.mutex.release()
    
    def image_captured(self, image):
        try:
            capture_fps = 1000.0 / (image.timestamp() - self.capture_last_ts)
        except:
            capture_fps = 0
        self.time_metrics_dict['cfps'] = capture_fps
        self.capture_last_ts = image.timestamp()

    def clear_all_dictionaries(self):
        """
        Clears the dictionary values
        """
        self.measurements_dict.clear()
        self.expressions_dict.clear()
        self.emotions_dict.clear()
        self.bounding_box_dict.clear()
        self.identities_dict.clear()
        self.gaze_metric_dict.clear()
        self.glasses_dict.clear()
        self.age_metric_dict.clear()
        self.age_category_dict.clear()
