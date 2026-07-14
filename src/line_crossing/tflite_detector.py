
import numpy as np
import cv2
from collections import deque


class SimpleTracker:
    """Lightweight object tracker using centroid distance."""
    def __init__(self, max_distance=50, max_age=30):
        self.next_track_id = 0
        self.tracks = {}  # {track_id: {'centroid': (x,y), 'age': 0, 'history': deque()}}
        self.max_distance = max_distance
        self.max_age = max_age

    def update(self, detections):
        """
        Args:
            detections: list of dicts with 'box', 'conf', 'class_id', 'centroid', 'foot_point'
        Returns:
            detections with 'track_id' added
        """
        # Get centroids from current detections
        current_centroids = np.array([det['centroid'] for det in detections]) if detections else np.empty((0, 2))

        # Get existing track centroids
        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid]['centroid'] for tid in track_ids]) if track_ids else np.empty((0, 2))

        # Assign detections to tracks
        assigned_tracks = set()
        assigned_detections = set()

        if len(track_centroids) > 0 and len(current_centroids) > 0:
            # Compute distance matrix
            dist_matrix = np.linalg.norm(track_centroids[:, np.newaxis] - current_centroids[np.newaxis, :], axis=2)

            # Greedy assignment
            while True:
                min_dist = np.min(dist_matrix, initial=np.inf)
                if min_dist > self.max_distance:
                    break
                track_idx, det_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                if track_idx in assigned_tracks or det_idx in assigned_detections:
                    dist_matrix[track_idx, det_idx] = np.inf
                    continue
                track_id = track_ids[track_idx]
                detections[det_idx]['track_id'] = track_id
                self.tracks[track_id]['centroid'] = detections[det_idx]['centroid']
                self.tracks[track_id]['foot_point'] = detections[det_idx]['foot_point']
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['history'].append(detections[det_idx]['centroid'])
                if len(self.tracks[track_id]['history']) > 10:
                    self.tracks[track_id]['history'].popleft()
                assigned_tracks.add(track_idx)
                assigned_detections.add(det_idx)
                dist_matrix[track_idx, :] = np.inf
                dist_matrix[:, det_idx] = np.inf

        # Create new tracks for unassigned detections
        for i, det in enumerate(detections):
            if i not in assigned_detections:
                det['track_id'] = self.next_track_id
                self.tracks[self.next_track_id] = {
                    'centroid': det['centroid'],
                    'foot_point': det['foot_point'],
                    'age': 0,
                    'history': deque([det['centroid']])
                }
                self.next_track_id += 1

        # Update age of unassigned tracks and remove old ones
        for tid in list(self.tracks.keys()):
            if tid not in [track_ids[i] for i in assigned_tracks]:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        return detections


class YOLOv8TFLite:
    """YOLOv8 TFLite inference with pre/post processing."""
    def __init__(self, model_path, img_size=640, conf_thresh=0.5, iou_thresh=0.45):
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.class_names = {0: 'person'}  # Only care about person for line crossing

        # Load TFLite model
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input quantization params
        self.input_scale = self.input_details[0]['quantization'][0]
        self.input_zero_point = self.input_details[0]['quantization'][1]
        self.input_dtype = self.input_details[0]['dtype']
        self.input_shape = self.input_details[0]['shape']

        print(f"Loaded TFLite model: {model_path}")
        print(f"Input shape: {self.input_shape}, dtype: {self.input_dtype}")

    def preprocess(self, img):
        """Resize, pad, and quantize image for TFLite input."""
        # Resize with aspect ratio preservation
        h, w = img.shape[:2]
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_w = self.img_size - new_w
        pad_h = self.img_size - new_h
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Convert RGB (YOLO expects RGB)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # Normalize and quantize
        input_data = rgb.astype(np.float32) / 255.0
        if self.input_dtype in [np.int8, np.uint8]:
            input_data = (input_data / self.input_scale + self.input_zero_point).astype(self.input_dtype)

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        return input_data, (scale, left, top, w, h)

    def postprocess(self, output, preproc_params, orig_shape):
        """Decode YOLOv8 TFLite output."""
        scale, pad_left, pad_top, orig_w, orig_h = preproc_params

        # YOLOv8 output shape: (1, 84, 8400) -> (batch, 4 + num_classes, num_predictions)
        output = output[0].T  # (8400, 84)

        # Extract boxes, scores, classes
        boxes = output[:, :4]  # (x_center, y_center, w, h)
        scores = output[:, 4:]  # class scores
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Filter by confidence and class (only person, class 0)
        mask = (confidences > self.conf_thresh) & (class_ids == 0)
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2)
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), self.conf_thresh, self.iou_thresh)
        if len(indices) == 0:
            return []
        indices = indices.flatten()

        # Scale boxes back to original image
        detections = []
        for i in indices:
            x1, y1, x2, y2 = boxes_xyxy[i]
            # Remove padding
            x1 -= pad_left
            y1 -= pad_top
            x2 -= pad_left
            y2 -= pad_top
            # Scale back to original size
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale
            # Clip to image boundaries
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            foot_x = (x1 + x2) / 2
            foot_y = y2

            detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'conf': float(confidences[i]),
                'class_id': int(class_ids[i]),
                'centroid': (int(cx), int(cy)),
                'foot_point': (int(foot_x), int(foot_y))
            })

        return detections

    def detect(self, img):
        """Run full detection pipeline."""
        orig_h, orig_w = img.shape[:2]
        input_data, preproc_params = self.preprocess(img)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Postprocess
        detections = self.postprocess(output_data, preproc_params, (orig_w, orig_h))
        return detections
