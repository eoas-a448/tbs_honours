import numpy as np
from cv2 import cv2

class MultiTrackerImproved:
    def __init__(self, tracker_type):
        self.trackers = []
        self.previous_states = []
        self.boxes = []
        self.tracker_type = tracker_type

    def update(self, img):
        trackers_to_delete = []
        i = 0

        for tracker in self.trackers:
            success, box = tracker.update(img)

            if not success and not self.previous_states[i]:
                trackers_to_delete.append(i)
            else:
                self.previous_states[i] = success
                if success:
                    self.boxes[i] = box
            
            i = i + 1

        boxes_np = np.empty((len(self.boxes),), dtype=object) # Because self.boxes is a list of tuples
        boxes_np[:] = self.boxes

        self.trackers = np.delete(self.trackers, trackers_to_delete).tolist()
        self.previous_states = np.delete(self.previous_states, trackers_to_delete).tolist()
        self.boxes = np.delete(boxes_np, trackers_to_delete).tolist()

    def add_tracker(self, img, rect):
        tracker = self.tracker_type()
        tracker.init(img, rect)

        self.trackers.append(tracker)
        self.previous_states.append(False)
        self.boxes.append(None)

    # This may add an extra loop iteration unnecessarily
    def get_boxes(self):
        boxes = []

        for box in self.boxes:
            if box is not None:
                boxes.append(box)

        return boxes