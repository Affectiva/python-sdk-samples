# !/usr/bin/env python3
from collections import defaultdict
from threading import Lock
import affvisionpy as af

class OccupantListener(af.OccupantListener):
    """
    OccupantListener class that return occupant metrics for processed frames.

    """

    def __init__(self, occupant_interval):
        super(OccupantListener, self).__init__()

        self.count = 0
        self.process_last_ts = 0.0
        self.occupant_interval = occupant_interval
        self.mutex = Lock()
        self.time_metrics_dict = defaultdict()
        self.occupants = defaultdict()
        self.confidence = defaultdict()
        self.bounding_box = defaultdict()
        self.regionId = defaultdict()
        self.region = defaultdict()
        self.regionType = defaultdict()

    def results_updated(self, occupants, frame):
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
        self.occupants = occupants
        self.clear_all_dictionaries()
        for occid, occ in occupants.items():
            bbox = occ.bounding_box
            self.bounding_box[occid] = [bbox.getTopLeft().x,
                                        bbox.getTopLeft().y,
                                        bbox.getBottomRight().x,
                                        bbox.getBottomRight().y]
            self.regionId[occid] = occ.matched_seat.cabin_region.id
            self.regionType[occid] = occ.matched_seat.cabin_region.type.name
            if self.regionId[occid] != -1:
                self.region[occid] = occ.matched_seat.cabin_region.vertices
            self.confidence[occid] = occ.matched_seat.match_confidence
        self.mutex.release()

    def get_callback_interval(self):
        return self.occupant_interval

    def clear_all_dictionaries(self):
        """
        Clears the dictionary values
        """
        self.confidence = defaultdict()
        self.bounding_box = defaultdict()
        self.regionId = defaultdict()
        self.region = defaultdict()
        self.regionType = defaultdict()
