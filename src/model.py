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

    def predict(self, frame_segment: np.ndarray, seg_h: int, seg_w: int, profiler=None):
        @contextlib.contextmanager
        def _stage(name):
            if profiler:
                with profiler.stage(name):
                    yield
            else:
                yield

        with _stage("preprocess"):
            input_tensor, scale, pad_x, pad_y = self.preprocessor.process(frame_segment)
        
        with _stage("inference"):
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        with _stage("postprocess"):
            result = self.postprocessor.process(
                output_data, (seg_h, seg_w), (self.in_h, self.in_w),
                self.scale_params, scale, pad_x, pad_y
            )
        return result
