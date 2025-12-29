"""
Visualization module for UI rendering.

Handles all visual output including skeleton drawing,
status panels, progress bars, and Chinese text rendering.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Dict, Optional
import platform


# COCO Skeleton connections for pose visualization
SKELETON_CONNECTIONS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]


class Visualizer:
    """
    Handles all visualization for the boxing action recognition system.
    
    Features:
    - Chinese text rendering with PIL
    - Pose skeleton drawing
    - Status panels with progress bars
    - Robot connection status indicator
    """
    
    def __init__(self):
        """Initialize visualizer with font settings."""
        self._font = None
        self._font_cache = {}
        self._load_font()
    
    def _load_font(self) -> None:
        """Load Chinese font for text rendering."""
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",           # macOS
            "/System/Library/Fonts/STHeiti Light.ttc",      # macOS fallback
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux
            "C:/Windows/Fonts/msyh.ttc",                    # Windows
        ]
        
        for path in font_paths:
            try:
                self._font = ImageFont.truetype(path, 40)
                return
            except:
                continue
        
        self._font = ImageFont.load_default()
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font at specific size with caching."""
        if size not in self._font_cache:
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "C:/Windows/Fonts/msyh.ttc",
            ]
            
            for path in font_paths:
                try:
                    self._font_cache[size] = ImageFont.truetype(path, size)
                    break
                except:
                    continue
            else:
                self._font_cache[size] = ImageFont.load_default()
        
        return self._font_cache[size]
    
    def put_chinese_text(
        self,
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_size: int = 40,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw Chinese text on image using PIL.
        
        Args:
            img: BGR format image
            text: Text content
            position: (x, y) coordinates
            font_size: Font size in pixels
            color: BGR color tuple
            
        Returns:
            Image with text drawn
        """
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = self._get_font(font_size)
        
        # Convert BGR to RGB for PIL
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        min_confidence: float = 0.3
    ) -> np.ndarray:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: BGR image
            keypoints: Array of shape (17, 3) with [x, y, confidence]
            color: Line color in BGR
            thickness: Line thickness
            min_confidence: Minimum confidence threshold
            
        Returns:
            Frame with skeleton drawn
        """
        # Draw skeleton lines
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if (keypoints[start_idx, 2] > min_confidence and 
                keypoints[end_idx, 2] > min_confidence):
                start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                cv2.line(frame, start_point, end_point, color, thickness)
        
        # Draw keypoints
        for i in range(17):
            if keypoints[i, 2] > min_confidence:
                x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        return frame
    
    def draw_person_bbox(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        confidence: float,
        is_selected: bool = False
    ) -> np.ndarray:
        """
        Draw person bounding box with label.
        
        Args:
            frame: BGR image
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence
            is_selected: Whether this is the selected target
            
        Returns:
            Frame with bounding box drawn
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        if is_selected:
            color = (255, 144, 30)  # Blue-ish for selected
            thickness = 3
            label = f"person {confidence:.2f} [ACTIVE]"
        else:
            color = (128, 128, 128)  # Gray for others
            thickness = 2
            label = f"person {confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame, label, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        return frame
    
    def draw_info_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        num_persons: int,
        robot_status: str,
        is_paused: bool = False
    ) -> np.ndarray:
        """
        Draw top information overlay.
        
        Args:
            frame: BGR image
            fps: Current FPS
            num_persons: Number of detected persons
            robot_status: "connected", "disconnected", or "disabled"
            is_paused: Whether recognition is paused
            
        Returns:
            Frame with overlay drawn
        """
        h, w = frame.shape[:2]
        
        # FPS display
        cv2.putText(
            frame, f'FPS: {fps:.1f}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        
        # Person count (green for single, orange for multiple)
        persons_color = (0, 255, 0) if num_persons == 1 else (0, 165, 255)
        cv2.putText(
            frame, f'Persons: {num_persons}', (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, persons_color, 2
        )
        
        # Robot status indicator (top right)
        if robot_status == "connected":
            robot_color = (0, 255, 0)
            status_text = "Robot: Connected"
        elif robot_status == "disconnected":
            robot_color = (0, 0, 255)
            status_text = "Robot: Disconnected"
        else:
            robot_color = (128, 128, 128)
            status_text = "Robot: Disabled"
        
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = w - text_size[0] - 40
        cv2.putText(
            frame, status_text, (text_x, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, robot_color, 2
        )
        
        # Status indicator circle
        cv2.circle(frame, (w - 20, 25), 8, robot_color, -1)
        cv2.circle(frame, (w - 20, 25), 8, (255, 255, 255), 1)
        
        # Pause indicator
        if is_paused:
            cv2.putText(
                frame, '[PAUSED]', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
        
        return frame
    
    def draw_recognition_panel(
        self,
        frame: np.ndarray,
        action_text: str,
        confidence_text: str,
        hint_bar_height: int = 35
    ) -> np.ndarray:
        """
        Draw left-bottom recognition panel.
        
        Args:
            frame: BGR image
            action_text: Current action name
            confidence_text: Confidence display text
            hint_bar_height: Height of bottom hint bar
            
        Returns:
            Frame with panel drawn
        """
        h, w = frame.shape[:2]
        
        if action_text == "No Person":
            # Small "No Person" panel
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (10, h - 60 - hint_bar_height),
                (200, h - 10 - hint_bar_height),
                (0, 0, 0), -1
            )
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            cv2.putText(
                frame, "No Person", (20, h - 30 - hint_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        else:
            # Full recognition panel
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (10, h - 155 - hint_bar_height),
                (350, h - 10 - hint_bar_height),
                (0, 0, 0), -1
            )
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            # Title
            cv2.putText(
                frame, "Recognition", (20, h - 130 - hint_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1
            )
            
            # Action name (Chinese)
            frame = self.put_chinese_text(
                frame, action_text,
                (20, h - 110 - hint_bar_height),
                font_size=32, color=(0, 255, 255)
            )
            
            # Confidence
            cv2.putText(
                frame, confidence_text, (20, h - 20 - hint_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
            )
        
        return frame
    
    def draw_status_panel(
        self,
        frame: np.ndarray,
        state: str,
        progress: int,
        is_paused: bool,
        is_cooling: bool,
        cooldown_remaining: float,
        confirmed_action: Optional[str],
        confirm_frames: int,
        confidence_threshold: float,
        show_progress_bar: bool = True,
        hint_bar_height: int = 35
    ) -> np.ndarray:
        """
        Draw right-bottom status panel.
        
        Args:
            frame: BGR image
            state: Current state (action name or "Standby")
            progress: Confirmation progress (0-100)
            is_paused: Whether recognition is paused
            is_cooling: Whether in cooldown period
            cooldown_remaining: Remaining cooldown time
            confirmed_action: Last confirmed action
            confirm_frames: Number of frames required
            confidence_threshold: Confidence threshold value
            show_progress_bar: Whether to show progress bar
            hint_bar_height: Height of bottom hint bar
            
        Returns:
            Frame with panel drawn
        """
        h, w = frame.shape[:2]
        
        state_box_width = 420
        state_box_height = 180 if show_progress_bar else 130
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (w - state_box_width - 10, h - state_box_height - 10 - hint_bar_height),
            (w - 10, h - 10 - hint_bar_height),
            (20, 20, 20), -1
        )
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        base_y = h - state_box_height - hint_bar_height
        
        # Title
        cv2.putText(
            frame, "Action Status",
            (w - state_box_width, base_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
        )
        
        # Parameters
        cv2.putText(
            frame,
            f"Frames: {confirm_frames} | Threshold: {confidence_threshold*100:.0f}%",
            (w - state_box_width, base_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1
        )
        
        # Progress bar
        if show_progress_bar:
            bar_width = state_box_width - 40
            bar_height = 16
            bar_x = w - state_box_width
            bar_y = base_y + 65
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (60, 60, 60), -1)
            
            # Fill
            if progress > 0:
                fill_width = int(bar_width * progress / 100)
                if progress >= 100:
                    bar_color = (0, 255, 0)
                elif progress >= 70:
                    bar_color = (0, 255, 255)
                else:
                    bar_color = (0, 165, 255)
                cv2.rectangle(frame, (bar_x, bar_y),
                             (bar_x + fill_width, bar_y + bar_height),
                             bar_color, -1)
            
            cv2.putText(frame, f"{progress}%",
                       (bar_x + bar_width + 5, bar_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            status_y_offset = 95
        else:
            status_y_offset = 65
        
        # Status display
        if is_paused:
            cv2.putText(
                frame, "[PAUSED]",
                (w - state_box_width, base_y + status_y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2
            )
            cv2.putText(
                frame, "Press SPACE to resume",
                (w - state_box_width, base_y + status_y_offset + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1
            )
        elif confirmed_action and progress >= 100:
            frame = self.put_chinese_text(
                frame, f"[OK] {confirmed_action}",
                (w - state_box_width, base_y + status_y_offset),
                font_size=36, color=(0, 255, 0)
            )
            if is_cooling:
                cv2.putText(
                    frame, f"Sent! Cooldown: {cooldown_remaining:.1f}s",
                    (w - state_box_width, base_y + status_y_offset + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
            else:
                cv2.putText(
                    frame, "Action Confirmed",
                    (w - state_box_width, base_y + status_y_offset + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
        elif is_cooling:
            cv2.putText(
                frame, "Cooling Down...",
                (w - state_box_width, base_y + status_y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2
            )
            cv2.putText(
                frame, f"Remaining: {cooldown_remaining:.1f}s",
                (w - state_box_width, base_y + status_y_offset + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1
            )
        elif state != "Standby":
            frame = self.put_chinese_text(
                frame, f"> {state}",
                (w - state_box_width, base_y + status_y_offset),
                font_size=36, color=(0, 255, 255)
            )
            cv2.putText(
                frame, "Confirming... Hold pose",
                (w - state_box_width, base_y + status_y_offset + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
        else:
            cv2.putText(
                frame, "Standby",
                (w - state_box_width, base_y + status_y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2
            )
            cv2.putText(
                frame, "Waiting for action...",
                (w - state_box_width, base_y + status_y_offset + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1
            )
        
        return frame
    
    def draw_hint_bar(
        self,
        frame: np.ndarray,
        robot_enabled: bool,
        robot_connected: bool
    ) -> np.ndarray:
        """
        Draw bottom hint bar with controls.
        
        Args:
            frame: BGR image
            robot_enabled: Whether robot is enabled
            robot_connected: Whether robot is connected
            
        Returns:
            Frame with hint bar drawn
        """
        h, w = frame.shape[:2]
        bar_height = 30
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        hint_text = "[Q] Quit  |  [SPACE] Pause"
        if robot_enabled:
            status = "Connected" if robot_connected else "Disconnected"
            hint_text += f"  |  Robot: {status}"
        
        cv2.putText(
            frame, hint_text, (15, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
        )
        
        return frame
    
    def draw_flash_border(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 8
    ) -> np.ndarray:
        """
        Draw flash border effect (for action confirmation).
        
        Args:
            frame: BGR image
            color: Border color in BGR
            thickness: Border thickness
            
        Returns:
            Frame with border drawn
        """
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)
        return frame
