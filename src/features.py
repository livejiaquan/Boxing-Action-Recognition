"""
Feature extraction module for pose-based action recognition.

Extracts 46-dimensional feature vector from 17 COCO keypoints,
focusing on upper body and arm movements critical for boxing actions.
"""

import numpy as np
from config import IMPORTANT_KEYPOINTS


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at point p2 formed by p1-p2-p3.
    
    Args:
        p1: First point [x, y]
        p2: Vertex point [x, y]
        p3: Third point [x, y]
        
    Returns:
        Angle in radians
    """
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle


def extract_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract 46-dimensional feature vector from normalized keypoints.
    
    Features include:
    - Joint coordinates (24 dim): x, y for 12 important keypoints
    - Wrist absolute positions (4 dim)
    - Arm extension distances (2 dim)
    - Wrist-shoulder relative positions (4 dim)
    - Wrist differences (3 dim)
    - Shoulder rotation angle (1 dim)
    - Elbow joint angles (2 dim)
    - Knee joint angles (2 dim)
    - Body center (hip center) (2 dim)
    - Wrist-to-hip distances (2 dim)
    
    Args:
        keypoints: Normalized keypoints array of shape (17, 3) [x, y, confidence]
        
    Returns:
        Feature vector of shape (46,)
    """
    features = []
    
    # Keypoint indices (COCO format)
    left_shoulder, right_shoulder = 5, 6
    left_elbow, right_elbow = 7, 8
    left_wrist, right_wrist = 9, 10
    left_hip, right_hip = 11, 12
    left_knee, right_knee = 13, 14
    left_ankle, right_ankle = 15, 16
    
    # [1] Joint coordinates (24 dim) - excluding head keypoints
    for i in IMPORTANT_KEYPOINTS:
        features.append(keypoints[i, 0])  # x
        features.append(keypoints[i, 1])  # y
    
    # [2] Wrist absolute positions (4 dim)
    features.append(keypoints[left_wrist, 0])
    features.append(keypoints[left_wrist, 1])
    features.append(keypoints[right_wrist, 0])
    features.append(keypoints[right_wrist, 1])
    
    # [3] Arm extension distances (2 dim)
    left_arm_ext = np.linalg.norm(
        keypoints[left_wrist, :2] - keypoints[left_shoulder, :2]
    )
    right_arm_ext = np.linalg.norm(
        keypoints[right_wrist, :2] - keypoints[right_shoulder, :2]
    )
    features.append(left_arm_ext)
    features.append(right_arm_ext)
    
    # [4] Wrist-shoulder relative positions (4 dim)
    left_wrist_rel = keypoints[left_wrist, :2] - keypoints[left_shoulder, :2]
    right_wrist_rel = keypoints[right_wrist, :2] - keypoints[right_shoulder, :2]
    features.append(left_wrist_rel[0])
    features.append(left_wrist_rel[1])
    features.append(right_wrist_rel[0])
    features.append(right_wrist_rel[1])
    
    # [5] Wrist differences (3 dim)
    wrist_y_diff = keypoints[left_wrist, 1] - keypoints[right_wrist, 1]
    wrist_x_diff = keypoints[left_wrist, 0] - keypoints[right_wrist, 0]
    arm_ext_diff = left_arm_ext - right_arm_ext
    features.append(wrist_y_diff)
    features.append(wrist_x_diff)
    features.append(arm_ext_diff)
    
    # [6] Shoulder rotation angle (1 dim)
    shoulder_vec = keypoints[right_shoulder, :2] - keypoints[left_shoulder, :2]
    shoulder_angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
    features.append(shoulder_angle)
    
    # [7] Elbow joint angles (2 dim)
    left_elbow_angle = calculate_angle(
        keypoints[left_shoulder, :2],
        keypoints[left_elbow, :2],
        keypoints[left_wrist, :2]
    )
    right_elbow_angle = calculate_angle(
        keypoints[right_shoulder, :2],
        keypoints[right_elbow, :2],
        keypoints[right_wrist, :2]
    )
    features.append(left_elbow_angle)
    features.append(right_elbow_angle)
    
    # [8] Knee joint angles (2 dim)
    left_knee_angle = calculate_angle(
        keypoints[left_hip, :2],
        keypoints[left_knee, :2],
        keypoints[left_ankle, :2]
    )
    right_knee_angle = calculate_angle(
        keypoints[right_hip, :2],
        keypoints[right_knee, :2],
        keypoints[right_ankle, :2]
    )
    features.append(left_knee_angle)
    features.append(right_knee_angle)
    
    # [9] Body center - hip center (2 dim)
    hip_center = (keypoints[left_hip, :2] + keypoints[right_hip, :2]) / 2
    features.append(hip_center[0])
    features.append(hip_center[1])
    
    # [10] Wrist-to-hip distances (2 dim)
    left_wrist_to_hip = np.linalg.norm(keypoints[left_wrist, :2] - hip_center)
    right_wrist_to_hip = np.linalg.norm(keypoints[right_wrist, :2] - hip_center)
    features.append(left_wrist_to_hip)
    features.append(right_wrist_to_hip)
    
    return np.array(features)  # Total: 46 dimensions


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints to body-relative coordinates.
    
    Normalizes x and y coordinates to [0, 1] range based on
    the bounding box of all keypoints.
    
    Args:
        keypoints: Raw keypoints array of shape (17, 3)
        
    Returns:
        Normalized keypoints array of shape (17, 3)
    """
    keypoints_norm = keypoints.copy()
    
    min_x, max_x = np.min(keypoints[:, 0]), np.max(keypoints[:, 0])
    min_y, max_y = np.min(keypoints[:, 1]), np.max(keypoints[:, 1])
    
    range_x = max_x - min_x if max_x > min_x else 1
    range_y = max_y - min_y if max_y > min_y else 1
    
    keypoints_norm[:, 0] = (keypoints[:, 0] - min_x) / range_x
    keypoints_norm[:, 1] = (keypoints[:, 1] - min_y) / range_y
    
    return keypoints_norm
