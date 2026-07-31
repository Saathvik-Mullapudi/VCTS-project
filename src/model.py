import logging
import numpy as np
import contextlib

logger = logging.getLogger(__name__)

# Try tflite-runtime first (for i.MX8)
try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
    NPU_AVAILABLE = True
    logger.info("Using tflite-runtime (i.MX8 mode)")
except ImportError:
    # Try TensorFlow (for Windows laptop)
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        # For TensorFlow >= 2.14, experimental.load_delegate is still valid
        try:
            load_delegate = tf.lite.experimental.load_delegate
        except AttributeError:
            load_delegate = None  # N/A on Windows
        NPU_AVAILABLE = False
        logger.info("Using TensorFlow for TFLite (Windows mode)")
    except ImportError as e:
        logger.error(f"Need either tensorflow or tflite-runtime installed! Error: {e}")
        exit(1)

from src.inference import CPUPreprocessor, OptimizedPostprocessor
from configs.settings import CONF_THRESHOLD, NMS_THRESHOLD, PERSON_CLASS_IDS

class TFLitePersonDetector:
    def __init__(self, model_path: str):
        logger.info(f"Loading: {model_path}")
        self.npu_status = "unknown"
        
        if NPU_AVAILABLE:
            try:
                delegate = load_delegate('libvx_delegate.so')
                self.interpreter = Interpreter(model_path, experimental_delegates=[delegate])
                self.npu_status = "loaded"
                logger.info("NPU delegate loaded")
            except Exception as e:
                self.npu_status = "fallback"
                logger.warning(f"NPU delegate failed, CPU fallback: {e}")
                self.interpreter = Interpreter(model_path, num_threads=4)
        else:
            self.npu_status = "unavailable"
            self.interpreter = Interpreter(model_path, num_threads=4)
            logger.info("NPU unavailable, using CPU")
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        input_shape = self.input_details[0]['shape']
        self.in_h, self.in_w = int(input_shape[1]), int(input_shape[2])
        in_dtype = self.input_details[0]['dtype']
        
        logger.info(f"Model input dtype: {in_dtype}, output dtype: {self.output_details[0]['dtype']}")
        if str(in_dtype) not in ("<class 'numpy.uint8'>", "uint8", "int8", "int32"):
            logger.warning("Model input does not look like a fully quantized int8/uint8 TFLite model")
        
        self.scale_params = {'scale': None, 'zero_point': 0}
        try:
            qp = self.output_details[0].get('quantization_parameters', {})
            if qp.get('scales'):
                self.scale_params['scale'] = float(qp['scales'][0])
            if qp.get('zero_points'):
                self.scale_params['zero_point'] = int(qp['zero_points'][0])
        except Exception as e:
            logger.debug(f"Could not read quantization params: {e}")
        
        self.preprocessor = CPUPreprocessor(self.in_w, self.in_h, in_dtype)
        self.postprocessor = OptimizedPostprocessor(
            conf_threshold=CONF_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            topk=500,
            class_ids=PERSON_CLASS_IDS
        )
        
        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_input, _, _, _ = self.preprocessor.process(dummy)
        self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
        self.interpreter.invoke()
        logger.info("Model ready")

    def _get_tile_bounds(self, display_w, display_h, line_start_x, tile_padding):
        """Right-half ROI with optional left padding (display coordinates)."""
        x0 = max(0, min(display_w // 2, line_start_x) - tile_padding)
        return x0, 0, display_w, display_h

    def _map_boxes_tile_to_full_ml(self, boxes, tile_x0, tile_y0, display_w, display_h):
        if not boxes:
            return boxes
        ml_scale_x = self.in_w / display_w
        ml_scale_y = self.in_h / display_h
        mapped = []
        for x1, y1, x2, y2 in boxes:
            dx1, dy1 = tile_x0 + x1, tile_y0 + y1
            dx2, dy2 = tile_x0 + x2, tile_y0 + y2
            mapped.append([
                int(dx1 * ml_scale_x), int(dy1 * ml_scale_y),
                int(dx2 * ml_scale_x), int(dy2 * ml_scale_y)
            ])
        return mapped

    def _map_boxes_tile_to_display(self, boxes, tile_x0, tile_y0):
        if not boxes:
            return boxes
        return [[int(tile_x0 + x1), int(tile_y0 + y1), int(tile_x0 + x2), int(tile_y0 + y2)] for x1, y1, x2, y2 in boxes]

    def _map_boxes_ml_to_display(self, boxes, display_w, display_h):
        if not boxes:
            return boxes
        scale_x = display_w / self.in_w
        scale_y = display_h / self.in_h
        return [[int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)] for x1, y1, x2, y2 in boxes]

    def predict(self, frame: np.ndarray, use_tiling: bool, tile_padding: int, line_start_x: int, profiler=None):
        """
        Accepts a full 1080p display frame, handles array slicing internally if tiling is enabled,
        runs the AI model, and maps the boxes back to display space.
        Returns: person_boxes (ML coords), person_boxes_display (Display coords), person_scores, person_classes
        """
        import contextlib
        @contextlib.contextmanager
        def _stage(name):
            if profiler:
                with profiler.stage(name): yield
            else: yield

        display_h, display_w = frame.shape[:2]
        
        if use_tiling:
            tile_x0, tile_y0, tile_x1, tile_y1 = self._get_tile_bounds(display_w, display_h, line_start_x, tile_padding)
            frame_segment = frame[tile_y0:tile_y1, tile_x0:tile_x1]
            seg_h, seg_w = frame_segment.shape[:2]
        else:
            frame_segment = frame
            seg_h, seg_w = display_h, display_w

        with _stage("preprocess"):
            input_tensor, scale, pad_x, pad_y = self.preprocessor.process(frame_segment)
        
        with _stage("inference"):
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        with _stage("postprocess"):
            raw_boxes, scores, classes = self.postprocessor.process(
                output_data, (seg_h, seg_w), (self.in_h, self.in_w),
                self.scale_params, scale, pad_x, pad_y
            )
            
        if use_tiling:
            boxes_display = self._map_boxes_tile_to_display(raw_boxes, tile_x0, tile_y0)
            boxes_ml = self._map_boxes_tile_to_full_ml(raw_boxes, tile_x0, tile_y0, display_w, display_h)
        else:
            boxes_display = raw_boxes
            boxes_ml = self._map_boxes_tile_to_full_ml(raw_boxes, 0, 0, display_w, display_h)
            
        return boxes_ml, boxes_display, scores, classes
