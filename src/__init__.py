"""
Core modules for Boxing Action Recognition System.
"""

from .features import extract_features
from .pose_validator import check_body_completeness, verify_crouch_action
from .visualizer import Visualizer
from .robot_client import RobotClient

__all__ = [
    "extract_features",
    "check_body_completeness",
    "verify_crouch_action",
    "Visualizer",
    "RobotClient",
]
