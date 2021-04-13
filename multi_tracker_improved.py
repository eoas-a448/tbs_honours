import numpy as np

def do_boxes_intersect(box1, box2):
    (x1, y1, w1, h1) = [int(v) for v in box1]
    (x2, y2, w2, h2) = [int(v) for v in box2]

    if(x1 >= (x2+w2) or x2 >= (x1+w1)): 
        return False

    if((y1+h1) <= y2 or (y2+h2) <= y1): 
        return False
  
    return True

class MultiTrackerImproved:
    def __init__(self, tracker_type):
        self.trackers = []
        self.previous_positions = []
        self.number_of_successes = []
        self.all_boxes = []
        self.pos_in_all_boxes = []
        self.confirmed_exhaust = []
        self.tracker_type = tracker_type

    def update(self, img, current_time_index):
        trackers_to_delete = []
        i = 0

        for tracker in self.trackers:
            success, box = tracker.update(img)
            is_deleted = False
            pos = self.pos_in_all_boxes[i]

            if success:
                self.number_of_successes[i] = self.number_of_successes[i] + 1
                self.all_boxes[pos][current_time_index] = box
                if self.number_of_successes[i] >= 3:
                    self.confirmed_exhaust[pos] = True
            else:
                trackers_to_delete.append(i)
                is_deleted = True
            
            (x, y, w, h) = [int(v) for v in box]

            if not is_deleted and (np.abs(x - self.previous_positions[i][0]) > 5 or np.abs(y - self.previous_positions[i][1]) > 5):
                trackers_to_delete.append(i)
                is_deleted = True
            
            self.previous_positions[i] = (x, y)
            
            i = i + 1

        previous_positions_np = np.empty((len(self.previous_positions),), dtype=object) # Because self.previous_positions is a list of tuples
        previous_positions_np[:] = self.previous_positions

        self.trackers = np.delete(self.trackers, trackers_to_delete).tolist()
        self.previous_positions = np.delete(previous_positions_np, trackers_to_delete).tolist()
        self.number_of_successes = np.delete(self.number_of_successes, trackers_to_delete).tolist()
        self.pos_in_all_boxes = np.delete(self.pos_in_all_boxes, trackers_to_delete).tolist()

    def add_tracker(self, img, rect, time_len):
        tracker = self.tracker_type()
        tracker.init(img, rect)

        self.trackers.append(tracker)
        self.previous_positions.append((rect[0], rect[1]))
        self.number_of_successes.append(0)
        self.confirmed_exhaust.append(False)
        self.all_boxes.append(np.empty(time_len, dtype=object))
        self.pos_in_all_boxes.append(len(self.all_boxes)-1)

    # Do this with numpy instead??
    def get_boxes(self, current_time_index):
        boxes = []
        boxes_final = []

        for i in range(len(self.all_boxes)):
            if self.confirmed_exhaust[i]:
                if self.all_boxes[i][current_time_index] is not None:
                    boxes.append(self.all_boxes[i][current_time_index])

        ##### System for checking if boxes overlap ########
        for box in boxes:
            for other_box in boxes:
                if box == other_box:
                    continue
                if do_boxes_intersect(box, other_box):
                    boxes_final.append(box)
                    break
        ###################################################

        return boxes_final
