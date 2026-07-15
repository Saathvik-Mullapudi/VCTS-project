# ============================================================================
# APPLICATION CONFIGURATION & CONSTANTS
# ============================================================================

# Model defaults
DEFAULT_MODEL = "models/yolov8n_saved_model/yolov8n_full_integer_quant.tflite"
DEFAULT_SOURCE = "data/videos/vid.mp4"
DEFAULT_OUTPUT = "outputs/videos/output.mp4"

# Resolution
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1920, 1080
ML_INPUT_WIDTH, ML_INPUT_HEIGHT = 640, 360
PREVIEW_WIDTH, PREVIEW_HEIGHT = 1280, 720

# Performance
SKIP_FRAMES = 2  # Process 1 frame, skip 2 (runs inference on every 3rd frame)

# Detection parameters
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

# Person class (COCO class 0)
PERSON_CLASS_IDS = [0]

# Tracking
MAX_DISAPPEARED = 15
MAX_TRACK_DISTANCE = 120

# Line crossing for data/videos/vid.mp4 (1920x1080 display space)
LINE_START = (691, 496)
LINE_END = (1593, 648)

# Tiling - run inference on the right half where the crossing line lives
USE_TILING = True
TILE_PADDING = 120  # pixels to extend the tile left of frame center

# Profiling
PROFILE_PRINT_EVERY = 30

# System monitoring
SYSTEM_MONITOR_INTERVAL = 1.0

# Windows fallback
WINDOWS_OUTPUT_CODEC = "mp4v"

# ============================================================================
# MAGIC NUMBERS REFACTORED
# ============================================================================

# Preprocessor constants
PREPROCESS_PAD_COLOR = 114        # Grey padding color used in YOLO preprocessing
PREPROCESS_INT8_OFFSET = 128      # Offset for int8 quantization

# Tracker constants
TRACKER_FOOT_POINT_MAX_DIST = 120 # Maximum pixel distance to associate a foot point with a tracked centroid

# GStreamer Pipeline parameters
GST_UDP_PORT = 5004               # Local UDP port for RTP streaming
GST_RTSP_PORT = "8554"            # Port for the RTSP server
GST_BITRATE_FILE = 8000           # H264 encoding bitrate for saving to disk
GST_BITRATE_STREAM = 5000         # H264 encoding bitrate for UDP/RTSP streaming
