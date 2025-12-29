"""
Pose validation module for quality control.

Provides validation functions to ensure pose data quality
before action classification, reducing false positives.
"""

import numpy as np
from typing import Tuple, List
from config import MIN_KEYPOINT_CONFIDENCE


def check_body_completeness(keypoints: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Check if essential body keypoints are detected.
    
    For boxing action recognition, we require at least one side of:
    - Shoulders (essential for arm movement reference)
    - Elbows (essential for punch detection)
    - Wrists (essential for punch detection)
    - Hips (essential for body posture)
    
    Args:
        keypoints: Array of shape (17, 3) with [x, y, confidence]
        
    Returns:
        Tuple of (is_complete, missing_parts):
            - is_complete: True if all required parts are detected
            - missing_parts: List of missing body part names
    """
    # COCO keypoint indices
    left_shoulder, right_shoulder = 5, 6
    left_elbow, right_elbow = 7, 8
    left_wrist, right_wrist = 9, 10
    left_hip, right_hip = 11, 12
    
    missing_parts = []
    
    # Check shoulders (at least one side)
    shoulder_ok = (
        keypoints[left_shoulder, 2] > MIN_KEYPOINT_CONFIDENCE or
        keypoints[right_shoulder, 2] > MIN_KEYPOINT_CONFIDENCE
    )
    if not shoulder_ok:
        missing_parts.append("肩膀")
    
    # Check elbows (at least one side)
    elbow_ok = (
        keypoints[left_elbow, 2] > MIN_KEYPOINT_CONFIDENCE or
        keypoints[right_elbow, 2] > MIN_KEYPOINT_CONFIDENCE
    )
    if not elbow_ok:
        missing_parts.append("手肘")
    
    # Check wrists (at least one side)
    wrist_ok = (
        keypoints[left_wrist, 2] > MIN_KEYPOINT_CONFIDENCE or
        keypoints[right_wrist, 2] > MIN_KEYPOINT_CONFIDENCE
    )
    if not wrist_ok:
        missing_parts.append("手腕")
    
    # Check hips (at least one side) - critical for body posture
    hip_ok = (
        keypoints[left_hip, 2] > MIN_KEYPOINT_CONFIDENCE or
        keypoints[right_hip, 2] > MIN_KEYPOINT_CONFIDENCE
    )
    if not hip_ok:
        missing_parts.append("髖部")
    
    is_complete = shoulder_ok and elbow_ok and wrist_ok and hip_ok
    
    return is_complete, missing_parts


def verify_crouch_action(keypoints: np.ndarray) -> bool:
    """
    Verify if a crouch action is genuine using body proportions.
    
    This function helps prevent false crouch detections caused by
    camera angles or partial body visibility. It analyzes:
    - Body ratio (lower body / upper body)
    - Knee position relative to hip-ankle line
    
    Args:
        keypoints: Raw (non-normalized) keypoints of shape (17, 3)
        
    Returns:
        True if the pose appears to be a genuine crouch
    """
    # COCO keypoint indices
    left_shoulder, right_shoulder = 5, 6
    left_hip, right_hip = 11, 12
    left_knee, right_knee = 13, 14
    left_ankle, right_ankle = 15, 16
    
    # Prerequisite: knees must be detected
    left_knee_conf = keypoints[left_knee, 2]
    right_knee_conf = keypoints[right_knee, 2]
    
    if left_knee_conf < 0.3 and right_knee_conf < 0.3:
        # Cannot determine crouch without visible knees
        return False
    
    # Calculate average Y positions
    shoulder_y = (keypoints[left_shoulder, 1] + keypoints[right_shoulder, 1]) / 2
    hip_y = (keypoints[left_hip, 1] + keypoints[right_hip, 1]) / 2
    knee_y = (keypoints[left_knee, 1] + keypoints[right_knee, 1]) / 2
    ankle_y = (keypoints[left_ankle, 1] + keypoints[right_ankle, 1]) / 2
    
    # Calculate body segment lengths
    upper_body = abs(hip_y - shoulder_y)   # Shoulder to hip
    lower_body = abs(ankle_y - hip_y)      # Hip to ankle
    
    # Avoid division by zero
    if lower_body < 10 or upper_body < 10:
        return False
    
    # Body ratio: normally ~1.2-1.8, becomes <1.0 when crouching
    body_ratio = lower_body / upper_body
    
    # Knee relative position: 0 = at hip level, 1 = at ankle level
    # When crouching, knee moves closer to hip (value < 0.4)
    if abs(ankle_y - hip_y) > 10:
        knee_relative_pos = abs(knee_y - hip_y) / abs(ankle_y - hip_y)
    else:
        knee_relative_pos = 0.5
    
    # Crouch criteria:
    # - Body ratio < 1.0 (shortened lower body)
    # - OR knee position < 0.4 (knee close to hip)
    is_crouch = (body_ratio < 1.0) or (knee_relative_pos < 0.4)
    
    return is_crouch
