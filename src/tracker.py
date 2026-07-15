import numpy as np
from collections import OrderedDict
from configs.settings import TRACKER_FOOT_POINT_MAX_DIST

# CENTROID TRACKER (FROM OVERSPEED PROJECT)
# ============================================================================
class CentroidTracker:
    def __init__(self, maxDisappeared=15, smoothing=0.75, maxDistance=120):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.crossed_status = {}
        self.smoothing = smoothing  # EMA factor: lower = smoother
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.crossed_status[self.nextObjectID] = {"side": None, "counted": False, "pending_frames": 0, "pending_side": None}
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.crossed_status:
            del self.crossed_status[objectID]

    def update(self, boxes):
        if len(boxes) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, []

        inputCentroids = np.zeros((len(boxes), 2), dtype="int")
        inputFootPoints = []
        for (i, box) in enumerate(boxes):
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, x2, y2 = box
            else:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids[i] = (cX, cY)
            footY = int(y2)
            inputFootPoints.append((cX, footY))

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                objectID = objectIDs[row]
                # EMA smoothing to reduce centroid jitter
                old = self.objects[objectID]
                new = inputCentroids[col]
                smoothed = (int(old[0] * (1 - self.smoothing) + new[0] * self.smoothing),
                            int(old[1] * (1 - self.smoothing) + new[1] * self.smoothing))
                self.objects[objectID] = smoothed
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects, inputFootPoints

    def get_foot_point(self, objectID, boxes):
        for (i, box) in enumerate(boxes):
            x1, y1, x2, y2 = box
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            if np.linalg.norm(np.array([cX, cY]) - np.array(self.objects[objectID])) < TRACKER_FOOT_POINT_MAX_DIST:
                return (cX, int(y2))
        return None

# ============================================================================
# LINE CROSSING COUNTER
# ============================================================================
class LineCrossingCounter:
    def __init__(self, line_start, line_end, ml_width, ml_height, display_width, display_height):
        self.line_start = line_start
        self.line_end = line_end
        self.ml_width = ml_width
        self.ml_height = ml_height
        self.display_width = display_width
        self.display_height = display_height
        self.crossing_count = 0

    def get_side(self, point, line_start, line_end):
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        return ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) > 0

    def get_distance(self, point, line_start, line_end):
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        num = abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1))
        den = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        return num / den if den > 0 else 0

    def update(self, tracked_objects, tracker, boxes):
        # Scale line from display coords to the active model input space.
        sx = self.ml_width / self.display_width
        sy = self.ml_height / self.display_height
        line_s = (self.line_start[0] * sx, self.line_start[1] * sy)
        line_e = (self.line_end[0]   * sx, self.line_end[1]   * sy)
        
        DEBOUNCE_FRAMES = 5  # Must be on new side for this many frames
        BUFFER_PIXELS = 20   # Spatial padding in ML coords (~60px in display space)
        
        for (objectID, centroid) in tracked_objects.items():
            foot_point = tracker.get_foot_point(objectID, boxes)
            if foot_point is None:
                continue
            
            current_side = self.get_side(foot_point, line_s, line_e)
            dist_to_line = self.get_distance(foot_point, line_s, line_e)
            
            state = tracker.crossed_status[objectID]
            
            if state["side"] is None:
                # First time seeing this object, just record the side
                state["side"] = current_side
                state["pending_frames"] = 0
                state["pending_side"] = None
                continue
            
            if state["counted"]:
                continue
                
            # If the centroid is inside the physical padding buffer, ignore it
            # This prevents jitter from crossing back and forth across the exact mathematical line
            if dist_to_line < BUFFER_PIXELS:
                continue
            
            if current_side != state["side"]:
                # Object appears to have crossed
                if state["pending_side"] == current_side:
                    state["pending_frames"] += 1
                else:
                    # Started crossing to a new side, reset counter
                    state["pending_side"] = current_side
                    state["pending_frames"] = 1
                
                if state["pending_frames"] >= DEBOUNCE_FRAMES:
                    # Confirmed crossing
                    self.crossing_count += 1
                    state["counted"] = True
                    state["side"] = current_side
                    print(f"[ALERT] ID{objectID} crossed! Total: {self.crossing_count}")
            else:
                # Back on original side, reset pending
                state["pending_frames"] = 0
                state["pending_side"] = None
        
        return self.crossing_count

