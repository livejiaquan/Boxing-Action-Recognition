"""
Configuration settings for Boxing Action Recognition System.

This module centralizes all configurable parameters for easy adjustment.
Modify these settings to tune the system behavior without touching the core code.
"""

# ==============================================================================
# Model Settings
# ==============================================================================

# YOLO Pose Estimation Model
# Set to True to use ONNX model (pruned), False for PyTorch model (auto-downloads)
USE_ONNX_MODEL = False
ONNX_MODEL_PATH = "models/best.onnx"           # ONNX model path (pruned version)
PT_MODEL_PATH = "yolo11n-pose.pt"              # PyTorch model (auto-downloads if missing)

# Action Classifier
CLASSIFIER_PATH = "models/boxing_classifier.pkl"

# ==============================================================================
# Recognition Parameters
# ==============================================================================

# Confidence threshold for action classification (0.0 - 1.0)
# Higher = more strict, fewer false positives but may miss some actions
# Recommended: 0.40 - 0.80
CONFIDENCE_THRESHOLD = 0.50

# Frame confirmation: number of consecutive frames required to confirm an action
# At 30 FPS: 5 frames ≈ 0.17s, 10 frames ≈ 0.33s
# Higher = more stable but slower response
CONFIRM_FRAMES = 10

# Prediction smoothing buffer size (voting mechanism)
# Uses majority voting over recent predictions
# Recommended: 3 - 7
PREDICTION_BUFFER_SIZE = 5

# Cooldown time after action confirmation (seconds)
# Prevents rapid repeated triggers of the same action
# Set to 0 to disable
COOLDOWN_TIME = 3.0

# ==============================================================================
# Body Completeness Validation
# ==============================================================================

# Minimum keypoint confidence to consider a joint detected
MIN_KEYPOINT_CONFIDENCE = 0.3

# Require full body (shoulders, elbows, wrists, hips) to be visible
# Set to False to allow partial body detection
REQUIRE_FULL_BODY = True

# ==============================================================================
# Display Settings
# ==============================================================================

# Show confirmation progress bar
SHOW_PROGRESS_BAR = True

# Show real-time recognition panel (bottom-left)
SHOW_RECOGNITION_PANEL = True

# Show action status panel (bottom-right)
SHOW_STATUS_PANEL = True

# Flash effect duration when action is sent (seconds)
FLASH_DURATION = 0.3

# ==============================================================================
# Robot Interface
# ==============================================================================

# Enable robot communication via TCP socket
ROBOT_ENABLED = True

# Robot connection settings
ROBOT_IP = "192.168.1.95"      # Robot IP address
ROBOT_HOST = ROBOT_IP          # Alias for compatibility
ROBOT_PORT = 50007             # Robot port
SOCKET_TIMEOUT = 3.0           # Connection timeout in seconds

# Auto-reconnection settings
ROBOT_RECONNECT_ENABLED = True
ROBOT_RECONNECT_INTERVAL = 5.0      # Seconds between reconnect attempts
ROBOT_RECONNECT_MAX_ATTEMPTS = 3    # Max attempts on startup

# ==============================================================================
# Action Signal Mapping
# ==============================================================================
# Maps action names (Chinese) to robot command signals
# None = action recognized but no signal sent (e.g., initial state)

ACTION_SIGNALS = {
    "初始狀態": None,       # Initial state - no signal, only resets
    "刺拳_左": "1",         # Boxing1: Left jab
    "刺拳_右": "2",         # Boxing2: Right jab
    "擺拳_左": "3",         # Boxing3: Left hook
    "擺拳_右": "4",         # Boxing4: Right hook
    "上勾拳_左": "5",       # Boxing5: Left uppercut
    "上勾拳_右": "6",       # Boxing6: Right uppercut
    "防禦姿態": "7",        # Boxing7: Guard position
    "蹲下": "8",            # Boxing8: Crouch/Duck
    "轉體_左": "9",         # Boxing9: Left pivot
    "轉體_右": "10",        # Boxing10: Right pivot
}

# ==============================================================================
# Action Classes Definition
# ==============================================================================
# Maps internal class codes to display names

CLASS_NAMES = {
    "00_stance_ready": "初始狀態",
    "01_stance_guard": "防禦姿態",
    "02_stance_crouch": "蹲下",
    "03_body_pivot_left": "轉體_左",
    "04_body_pivot_right": "轉體_右",
    "05_punch_jab_left": "刺拳_左",
    "06_punch_jab_right": "刺拳_右",
    "07_punch_hook_left": "擺拳_左",
    "08_punch_hook_right": "擺拳_右",
    "09_punch_upper_left": "上勾拳_左",
    "10_punch_upper_right": "上勾拳_右",
}

# Class order for training/evaluation
CLASS_ORDER = list(CLASS_NAMES.keys())

# ==============================================================================
# COCO Keypoint Definition
# ==============================================================================

COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# Important keypoints for feature extraction (excluding head)
IMPORTANT_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
