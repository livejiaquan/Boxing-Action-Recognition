#!/usr/bin/env python3
"""
Boxing Action Recognition System

Real-time boxing action recognition using YOLO11-pose and Random Forest.
Supports multi-person detection with automatic target selection,
frame confirmation mechanism, and optional robot integration.

Usage:
    python main.py

Controls:
    q       - Quit
    SPACE   - Pause/Resume recognition
"""

import cv2
import numpy as np
import joblib
import time
import platform
from collections import Counter
from typing import Optional, Dict, Any

from config import (
    # Model settings
    USE_ONNX_MODEL, ONNX_MODEL_PATH, PT_MODEL_PATH,
    CLASSIFIER_PATH,
    # Recognition parameters
    CONFIDENCE_THRESHOLD, CONFIRM_FRAMES,
    PREDICTION_BUFFER_SIZE, COOLDOWN_TIME,
    # Display settings
    SHOW_PROGRESS_BAR, SHOW_RECOGNITION_PANEL, SHOW_STATUS_PANEL,
    # Robot settings
    ROBOT_ENABLED, ROBOT_IP, ROBOT_PORT,
    ROBOT_RECONNECT_ENABLED, ROBOT_RECONNECT_INTERVAL,
    ROBOT_RECONNECT_MAX_ATTEMPTS,
    # Validation settings
    REQUIRE_FULL_BODY,
    # Mappings
    CLASS_NAMES, ACTION_SIGNALS
)
from src.features import extract_features, normalize_keypoints
from src.pose_validator import check_body_completeness, verify_crouch_action
from src.visualizer import Visualizer
from src.robot_client import RobotClient


def find_camera() -> int:
    """
    Find available camera index.
    
    On macOS, scans for available cameras to avoid selecting
    iPhone Continuity Camera by mistake.
    
    Returns:
        Camera index (default 0)
    """
    if platform.system() != 'Darwin':
        return 0
    
    print("\n[INFO] Scanning available cameras...")
    available_cameras = []
    
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            backend_name = cap.getBackendName()
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            ret, frame = cap.read()
            if ret:
                available_cameras.append({
                    'index': i,
                    'backend': backend_name,
                    'resolution': f"{int(width)}x{int(height)}"
                })
                print(f"  [Camera {i}] {backend_name} ({int(width)}x{int(height)})")
            
            cap.release()
    
    if not available_cameras:
        print("[WARNING] No camera found, using default index 0")
        return 0
    
    selected_index = available_cameras[0]['index']
    
    if len(available_cameras) > 1:
        print(f"\n[INFO] Found {len(available_cameras)} cameras")
        print(f"[INFO] Selected camera {selected_index} (typically built-in)")
    else:
        print(f"[INFO] Using camera {selected_index}")
    
    return selected_index


def select_target_person(boxes, frame_width: int, frame_height: int) -> tuple:
    """
    Select primary person from multiple detections.
    
    Strategy: Prefer high confidence (>0.5) + closest to frame center.
    
    Args:
        boxes: YOLO detection boxes
        frame_width: Frame width
        frame_height: Frame height
        
    Returns:
        Tuple of (selected_index, person_candidates)
    """
    num_persons = len(boxes)
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    person_candidates = []
    for idx in range(num_persons):
        box = boxes[idx]
        box_conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        
        # Calculate bounding box center
        box_center_x = (xyxy[0] + xyxy[2]) / 2
        box_center_y = (xyxy[1] + xyxy[3]) / 2
        
        # Calculate distance to frame center
        distance = np.sqrt(
            (box_center_x - frame_center_x)**2 +
            (box_center_y - frame_center_y)**2
        )
        
        person_candidates.append({
            'idx': idx,
            'confidence': box_conf,
            'bbox': xyxy,
            'center': (box_center_x, box_center_y),
            'distance_to_center': distance
        })
    
    # Selection: high confidence + closest to center
    high_conf = [p for p in person_candidates if p['confidence'] > 0.5]
    
    if high_conf:
        selected = min(high_conf, key=lambda x: x['distance_to_center'])
    elif person_candidates:
        selected = max(person_candidates, key=lambda x: x['confidence'])
    else:
        return -1, person_candidates
    
    return selected['idx'], person_candidates


def main():
    """Main entry point for the boxing action recognition system."""
    
    # === Import YOLO (delayed to show startup message first) ===
    print("=" * 60)
    print("Boxing Action Recognition System")
    print("=" * 60)
    
    from ultralytics import YOLO
    import os
    
    # === Load YOLO pose model ===
    if USE_ONNX_MODEL and os.path.exists(ONNX_MODEL_PATH):
        print(f"\n[INFO] Loading ONNX model: {ONNX_MODEL_PATH}")
        model = YOLO(ONNX_MODEL_PATH, task='pose')
        model_type = "ONNX"
    else:
        print(f"\n[INFO] Loading PyTorch model: {PT_MODEL_PATH}")
        model = YOLO(PT_MODEL_PATH)
        model_type = "PyTorch"
    print("[OK] Pose model loaded successfully")
    
    # === Load classifier ===
    print("\n[INFO] Loading action classifier...")
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"[ERROR] {CLASSIFIER_PATH} not found")
        print("Please run: python scripts/train.py")
        return
    
    model_data = joblib.load(CLASSIFIER_PATH)
    classifier = model_data['classifier']
    class_names = model_data['class_names']
    print("[OK] Classifier loaded successfully")
    
    # === Initialize visualizer ===
    viz = Visualizer()
    
    # === Initialize robot client (optional) ===
    robot: Optional[RobotClient] = None
    if ROBOT_ENABLED:
        print(f"\n[INFO] Connecting to robot at {ROBOT_IP}:{ROBOT_PORT}...")
        robot = RobotClient(
            ip=ROBOT_IP,
            port=ROBOT_PORT,
            auto_reconnect=ROBOT_RECONNECT_ENABLED,
            reconnect_interval=ROBOT_RECONNECT_INTERVAL,
            max_reconnect_attempts=ROBOT_RECONNECT_MAX_ATTEMPTS
        )
        
        # Try initial connection
        if robot.connect():
            print("[OK] Robot connected successfully")
        else:
            print("[INFO] Robot not available, continuing without robot")
    
    # === Open webcam ===
    print("\n[INFO] Starting webcam...")
    camera_index = find_camera()
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        if camera_index != 0:
            print(f"[WARNING] Camera {camera_index} unavailable, trying index 0...")
            cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Failed to open webcam")
        if robot:
            robot.disconnect()
        return
    
    print(f"[OK] Camera {camera_index} opened successfully")
    
    # === Create window ===
    window_name = 'Boxing Action Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # === Print configuration ===
    print("\n" + "=" * 60)
    print("[OK] System started successfully")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  - Model: {model_type}")
    print(f"  - Platform: {platform.system()}")
    print(f"  - Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  - Confirmation frames: {CONFIRM_FRAMES}")
    print(f"  - Prediction buffer: {PREDICTION_BUFFER_SIZE}")
    print(f"  - Cooldown time: {COOLDOWN_TIME}s")
    print(f"  - Robot: {'Connected' if robot and robot.connected else 'Disabled'}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'SPACE' to toggle pause")
    print("=" * 60 + "\n")
    
    # === State variables ===
    prev_time = time.time()
    prediction_buffer = []
    confirm_buffer = []
    last_sent_action = None
    last_sent_time = 0
    is_cooling_down = False
    current_state = "Standby"
    current_state_confidence = 0.0
    confirmed_action = None
    confirm_progress = 0
    is_paused = False
    flash_frame = False
    flash_start_time = 0
    FLASH_DURATION = 0.3
    
    # === Main loop ===
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            h, w = frame.shape[:2]
            
            # Cooldown check
            if COOLDOWN_TIME > 0 and is_cooling_down:
                if time.time() - last_sent_time >= COOLDOWN_TIME:
                    is_cooling_down = False
                    print("[INFO] Cooldown complete, ready for next action")
            
            # YOLO inference
            results = model(frame, verbose=False, half=False, imgsz=640)
            annotated_frame = frame.copy()
            
            # Initialize display variables
            action_text = "No Person"
            confidence_text = ""
            confidence = 0.0
            selected_idx = -1
            person_candidates = []
            
            # === Person detection and selection ===
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                selected_idx, person_candidates = select_target_person(
                    results[0].boxes, w, h
                )
                
                # Draw bounding boxes
                for person in person_candidates:
                    annotated_frame = viz.draw_person_bbox(
                        annotated_frame,
                        person['bbox'],
                        person['confidence'],
                        is_selected=(person['idx'] == selected_idx)
                    )
            
            # === Keypoint analysis for selected person ===
            if selected_idx >= 0 and results[0].keypoints is not None:
                all_keypoints = results[0].keypoints.data.cpu().numpy()
                
                if selected_idx < len(all_keypoints):
                    keypoints = all_keypoints[selected_idx]
                    
                    # Draw skeleton
                    annotated_frame = viz.draw_skeleton(annotated_frame, keypoints)
                    
                    # Skip classification if paused
                    if not is_paused:
                        # Body completeness check
                        body_ready = True
                        if REQUIRE_FULL_BODY:
                            body_complete, missing = check_body_completeness(keypoints)
                            if not body_complete:
                                body_ready = False
                                action_text = "等待入鏡..."
                                confidence_text = f"缺少: {', '.join(missing)}"
                                prediction_buffer.clear()
                                confirm_buffer.clear()
                                confirm_progress = 0
                                current_state = "Standby"
                                current_state_confidence = 0.0
                        
                        if body_ready:
                            # Normalize keypoints
                            keypoints_norm = normalize_keypoints(keypoints)
                            
                            # Extract features
                            features = extract_features(keypoints_norm)
                            
                            # Predict action
                            prediction = classifier.predict([features])[0]
                            probabilities = classifier.predict_proba([features])[0]
                            confidence = np.max(probabilities)
                            
                            # Prediction smoothing (buffer voting)
                            prediction_buffer.append(prediction)
                            if len(prediction_buffer) > PREDICTION_BUFFER_SIZE:
                                prediction_buffer.pop(0)
                            
                            if len(prediction_buffer) >= 3:
                                most_common = Counter(prediction_buffer).most_common(1)[0][0]
                                final_prediction = most_common
                            else:
                                final_prediction = prediction
                            
                            # Convert to action name
                            if isinstance(final_prediction, str) and final_prediction in CLASS_NAMES.values():
                                action_text = final_prediction
                            else:
                                action_text = CLASS_NAMES.get(final_prediction, str(final_prediction))
                            
                            # Crouch verification
                            if action_text == "蹲下":
                                if not verify_crouch_action(keypoints):
                                    action_text = "防禦姿態"
                                    confidence = confidence * 0.8
                            
                            confidence_text = f"Conf: {confidence*100:.1f}%"
                            
                            # Frame confirmation mechanism
                            if confidence >= CONFIDENCE_THRESHOLD and not is_cooling_down:
                                confirm_buffer.append(action_text)
                                if len(confirm_buffer) > CONFIRM_FRAMES:
                                    confirm_buffer.pop(0)
                                
                                if len(confirm_buffer) >= CONFIRM_FRAMES:
                                    if len(set(confirm_buffer)) == 1:
                                        confirmed_action = confirm_buffer[0]
                                        confirm_progress = 100
                                        
                                        if confirmed_action != last_sent_action:
                                            if confirmed_action == "初始狀態":
                                                print(f"[RESET] Back to initial state")
                                                last_sent_action = confirmed_action
                                            else:
                                                print(f"[ACTION] Confirmed: {confirmed_action}")
                                                last_sent_action = confirmed_action
                                                last_sent_time = time.time()
                                                
                                                if COOLDOWN_TIME > 0:
                                                    is_cooling_down = True
                                                
                                                # Send to robot
                                                if robot and robot.connected:
                                                    if robot.send_action(confirmed_action):
                                                        flash_frame = True
                                                        flash_start_time = time.time()
                                    else:
                                        most_common_in_buffer = Counter(confirm_buffer).most_common(1)[0]
                                        confirm_progress = int(most_common_in_buffer[1] / CONFIRM_FRAMES * 100)
                                else:
                                    confirm_progress = int(len(confirm_buffer) / CONFIRM_FRAMES * 100)
                                
                                current_state = action_text
                                current_state_confidence = confidence
                            else:
                                if confidence < CONFIDENCE_THRESHOLD:
                                    confirm_buffer.clear()
                                    confirm_progress = 0
                                    if action_text != current_state:
                                        last_sent_action = None
                                
                                if not is_cooling_down:
                                    current_state = "Standby"
                                    current_state_confidence = 0.0
            
            # === Calculate FPS ===
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # === Draw UI ===
            num_persons = len(results[0].boxes) if results[0].boxes is not None else 0
            robot_status = "disabled"
            if ROBOT_ENABLED:
                robot_status = "connected" if robot and robot.connected else "disconnected"
            
            annotated_frame = viz.draw_info_overlay(
                annotated_frame, fps, num_persons, robot_status, is_paused
            )
            
            if SHOW_RECOGNITION_PANEL:
                annotated_frame = viz.draw_recognition_panel(
                    annotated_frame, action_text, confidence_text
                )
            
            if SHOW_STATUS_PANEL:
                cooldown_remaining = max(0, COOLDOWN_TIME - (time.time() - last_sent_time))
                annotated_frame = viz.draw_status_panel(
                    annotated_frame,
                    current_state, confirm_progress,
                    is_paused, is_cooling_down, cooldown_remaining,
                    confirmed_action, CONFIRM_FRAMES, CONFIDENCE_THRESHOLD,
                    SHOW_PROGRESS_BAR
                )
            
            annotated_frame = viz.draw_hint_bar(
                annotated_frame, ROBOT_ENABLED,
                robot.connected if robot else False
            )
            
            # Flash effect
            if flash_frame:
                if time.time() - flash_start_time < FLASH_DURATION:
                    annotated_frame = viz.draw_flash_border(annotated_frame)
                else:
                    flash_frame = False
            
            # Display frame
            cv2.imshow(window_name, annotated_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Exiting program...")
                break
            elif key == ord(' '):
                is_paused = not is_paused
                print(f"[INFO] Recognition {'PAUSED' if is_paused else 'RESUMED'}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if robot:
            robot.disconnect()
        print("[OK] Program terminated")


if __name__ == "__main__":
    main()
