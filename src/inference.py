import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)
from configs.settings import PREPROCESS_PAD_COLOR, PREPROCESS_INT8_OFFSET

# ============================================================================
# CPU PREPROCESSOR 
# ============================================================================
class CPUPreprocessor:
    def __init__(self, target_width, target_height, input_dtype):
        self.target_width = target_width
        self.target_height = target_height
        self.input_dtype = input_dtype
        self.output_buffer = np.zeros((1, target_height, target_width, 3), dtype=input_dtype)
        
    def process(self, frame):
        orig_h, orig_w = frame.shape[:2]
        target_h, target_w = self.target_height, self.target_width
        
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        if new_w != orig_w or new_h != orig_h:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        padded = np.full((target_h, target_w, 3), PREPROCESS_PAD_COLOR, dtype=np.uint8)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = frame
        
        if self.input_dtype == np.uint8:
            np.copyto(self.output_buffer[0], padded)
        elif self.input_dtype == np.int8:
            np.subtract(padded, PREPROCESS_INT8_OFFSET, out=self.output_buffer[0], casting='unsafe')
        else:
            self.output_buffer[0] = padded.astype(np.float32) / 255.0
            
        return self.output_buffer, scale, pad_x, pad_y

# ============================================================================
# OPTIMIZED POSTPROCESSOR (FROM OVERSPEED PROJECT)
# ============================================================================
class OptimizedPostprocessor:
    def __init__(self, conf_threshold=0.3, nms_threshold=0.45, topk=500, class_ids=None):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.topk = topk
        self.class_ids = class_ids or [0]
    
    def _nms_numpy(self, boxes, scores, iou_threshold):
        if boxes.shape[0] == 0:
            return np.array([], dtype=int)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=int)
    
    def process(self, output_data, frame_shape, input_shape, scale_params, scale, pad_x, pad_y):
        out = output_data
        if scale_params['scale'] is not None:
            out = (out.astype(np.float32) - scale_params['zero_point']) * scale_params['scale']
        
        out = np.squeeze(out)
        if out.ndim == 2 and out.shape[0] < out.shape[1]:
            out = out.T
        
        if out.ndim == 1:
            out = np.expand_dims(out, 0)
        if out.shape[0] == 0 or out.shape[1] < 5:
            return [], [], []
        
        frame_h, frame_w = frame_shape
        in_h, in_w = input_shape
        
        class_scores = out[:, 4:]
        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
        
        conf_mask = max_scores > self.conf_threshold
        if not np.any(conf_mask):
            return [], [], []
        
        valid_indices = np.where(conf_mask)[0]
        valid_indices = valid_indices[np.isin(class_ids[valid_indices], self.class_ids)]
        if len(valid_indices) == 0:
            return [], [], []

        sel_scores = max_scores[valid_indices]
        sel_cls = class_ids[valid_indices]
        sel_preds = out[valid_indices, :4]
        
        if sel_scores.shape[0] > self.topk:
            topk_inds = np.argpartition(-sel_scores, self.topk)[:self.topk]
            sel_scores = sel_scores[topk_inds]
            sel_cls = sel_cls[topk_inds]
            sel_preds = sel_preds[topk_inds]
        
        # Explicit Data Contract: YOLOv8 outputs are expected in model pixel coordinates (0 to in_w)
        # No guessing scales. Just apply deterministic affine transform back to the frame.


        # Deterministic check for coordinate normalization:
        # 1. Quantized models output dequantized coordinates in [0, 2] range (normalized).
        # 2. Float32 models output coordinates in pixel space [0, 640] directly.
        # Fallback check (max_val <= 2.0) covers any float32 model exported with normalized output.
        is_normalized = (scale_params['scale'] is not None) or (float(np.nanmax(sel_preds[:, [0, 2]])) <= 2.0)
        
        if is_normalized:
            cx = (sel_preds[:, 0] * in_w - pad_x) / scale
            cy = (sel_preds[:, 1] * in_h - pad_y) / scale
            bw = (sel_preds[:, 2] * in_w) / scale
            bh = (sel_preds[:, 3] * in_h) / scale
        else:
            cx = (sel_preds[:, 0] - pad_x) / scale
            cy = (sel_preds[:, 1] - pad_y) / scale
            bw = sel_preds[:, 2] / scale
            bh = sel_preds[:, 3] / scale
        
        x1 = np.clip(cx - bw / 2.0, 0, frame_w - 1)
        y1 = np.clip(cy - bh / 2.0, 0, frame_h - 1)
        x2 = np.clip(cx + bw / 2.0, 0, frame_w - 1)
        y2 = np.clip(cy + bh / 2.0, 0, frame_h - 1)
        
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        keep = self._nms_numpy(boxes, sel_scores.astype(np.float32), self.nms_threshold)
        
        if keep.size == 0:
            return [], [], []
        
        return boxes[keep].astype(np.int32).tolist(), sel_scores[keep].tolist(), sel_cls[keep].tolist()

# ============================================================================
