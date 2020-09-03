# !/usr/bin/env python3
from collections import defaultdict
from threading import Lock
import affvisionpy as af

class ObjectListener(af.ObjectListener):
    """
    ObjectListener class that return object metrics for processed frames.

    """

    def __init__(self, object_interval):
        super(ObjectListener, self).__init__()

        self.count = 0
        self.process_last_ts = 0.0
        self.mutex = Lock()
        self.objects = defaultdict()
        self.confidence = defaultdict()
        self.bounding_box = defaultdict()
        self.region = defaultdict()
        self.regionId = defaultdict()
        self.regionConfidence = defaultdict()
        self.regionType = defaultdict()
        self.type = defaultdict()
        self.callback = defaultdict()
        self.callback[af.Feature.phones] = object_interval
        self.callback[af.Feature.child_seats] = object_interval

    def results_updated(self, objects, frame):
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
        self.objects = objects
        self.clear_all_dictionaries()
        for oid, obj in objects.items():
            bbox = obj.bounding_box
            self.bounding_box[oid] = [bbox.getTopLeft().x,
                                      bbox.getTopLeft().y,
                                      bbox.getBottomRight().x,
                                      bbox.getBottomRight().y]
            self.type[oid] = obj.type
            self.confidence[oid] = obj.confidence
            self.regionId[oid] = obj.matched_regions[0].cabin_region.id
            self.regionConfidence[oid] = obj.matched_regions[0].match_confidence
            self.regionType[oid] = obj.matched_regions[0].cabin_region.type.name
            if self.regionId[oid] != -1:
                self.region[oid] = obj.matched_regions[0].cabin_region.vertices
        self.mutex.release()

    def get_callback_interval(self):
        return self.callback

    def clear_all_dictionaries(self):
        """
        Clears the dictionary values
        """
        self.confidence.clear()
        self.bounding_box.clear()
        self.region.clear()
        self.regionId.clear()
        self.regionConfidence.clear()
        self.regionType.clear()
        self.type.clear()
