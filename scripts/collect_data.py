#!/usr/bin/env python3
"""
Collect training data for boxing action classifier.

This script captures pose keypoints from webcam and saves them
with action labels for training the classifier.

Usage:
    python scripts/collect_data.py

Controls:
    0-9     - Label and save current pose
    s       - Save data to JSON
    q       - Quit
"""

import cv2
import numpy as np
import json
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASS_ORDER, CLASS_NAMES, PT_MODEL_PATH


def main():
    from ultralytics import YOLO
    
    print("=" * 60)
    print("Boxing Action Data Collection")
    print("=" * 60)
    
    # Load YOLO model
    print("\n[INFO] Loading YOLO pose model...")
    model = YOLO(PT_MODEL_PATH)
    print("[OK] Model loaded")
    
    # Open webcam
    print("\n[INFO] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Failed to open webcam")
        return
    
    # Output file
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "keypoints_data.json")
    
    # Load existing data if available
    collected_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            collected_data = json.load(f)
        print(f"[INFO] Loaded {len(collected_data)} existing samples")
    
    # Display class options
    print("\n" + "=" * 60)
    print("Press number key to label current pose:")
    print("=" * 60)
    for i, class_name in enumerate(CLASS_ORDER):
        display_name = CLASS_NAMES.get(class_name, class_name)
        print(f"  [{i}] {class_name} ({display_name})")
    print("\n  [s] Save data to file")
    print("  [q] Quit")
    print("=" * 60 + "\n")
    
    window_name = "Data Collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    current_keypoints = None
    last_save_time = 0
    save_cooldown = 0.5  # Minimum time between saves
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # YOLO inference
        results = model(frame, verbose=False, imgsz=640)
        annotated_frame = frame.copy()
        
        # Draw detections
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Get first person's keypoints
            if results[0].keypoints is not None:
                keypoints = results[0].keypoints.data[0].cpu().numpy()
                current_keypoints = keypoints.tolist()
                
                # Draw skeleton
                skeleton = [
                    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
                    [7, 9], [8, 10]
                ]
                
                for start_idx, end_idx in skeleton:
                    if keypoints[start_idx, 2] > 0.3 and keypoints[end_idx, 2] > 0.3:
                        start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                        end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                        cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)
                
                for i in range(17):
                    if keypoints[i, 2] > 0.3:
                        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
                        cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)
        else:
            current_keypoints = None
        
        # Display info
        cv2.putText(
            annotated_frame, f"Samples: {len(collected_data)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        
        status = "Ready" if current_keypoints else "No Person"
        status_color = (0, 255, 0) if current_keypoints else (0, 0, 255)
        cv2.putText(
            annotated_frame, f"Status: {status}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
        )
        
        # Instructions
        cv2.putText(
            annotated_frame, "Press 0-9 to label pose | s=save | q=quit",
            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        
        cv2.imshow(window_name, annotated_frame)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save data
            with open(output_file, 'w') as f:
                json.dump(collected_data, f, indent=2)
            print(f"[OK] Saved {len(collected_data)} samples to {output_file}")
        elif key >= ord('0') and key <= ord('9') and key - ord('0') < len(CLASS_ORDER):
            # Label current pose
            if current_keypoints and time.time() - last_save_time > save_cooldown:
                class_idx = key - ord('0')
                class_name = CLASS_ORDER[class_idx]
                
                collected_data.append({
                    'class': class_name,
                    'keypoints': current_keypoints,
                    'timestamp': datetime.now().isoformat()
                })
                
                last_save_time = time.time()
                display_name = CLASS_NAMES.get(class_name, class_name)
                print(f"[+] Added: {class_name} ({display_name}) - Total: {len(collected_data)}")
    
    # Final save
    if collected_data:
        with open(output_file, 'w') as f:
            json.dump(collected_data, f, indent=2)
        print(f"\n[OK] Final save: {len(collected_data)} samples")
    
    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Data collection complete")


if __name__ == "__main__":
    main()
