"""
Robot communication client module.

Handles TCP socket connection to robot controller
with automatic reconnection and status management.
"""

import socket
import time
import threading
from typing import Optional, Callable
from config import (
    ROBOT_IP, 
    ROBOT_PORT, 
    SOCKET_TIMEOUT,
    ROBOT_RECONNECT_ENABLED,
    ROBOT_RECONNECT_INTERVAL,
    ROBOT_RECONNECT_MAX_ATTEMPTS,
    ACTION_SIGNALS
)


class RobotClient:
    """
    Manages TCP socket connection to robot controller.
    
    Features:
    - Thread-safe connection management
    - Automatic reconnection with configurable retry
    - Connection status callbacks
    - Action signal mapping
    """
    
    def __init__(
        self,
        ip: str = ROBOT_IP,
        port: int = ROBOT_PORT,
        timeout: float = SOCKET_TIMEOUT,
        auto_reconnect: bool = ROBOT_RECONNECT_ENABLED,
        reconnect_interval: float = ROBOT_RECONNECT_INTERVAL,
        max_reconnect_attempts: int = ROBOT_RECONNECT_MAX_ATTEMPTS,
        on_status_change: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize robot client.
        
        Args:
            ip: Robot IP address
            port: Robot TCP port
            timeout: Socket timeout in seconds
            auto_reconnect: Whether to automatically reconnect
            reconnect_interval: Seconds between reconnect attempts
            max_reconnect_attempts: Maximum reconnection attempts (-1 for infinite)
            on_status_change: Callback for connection status changes
        """
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.on_status_change = on_status_change
        
        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._reconnecting = False
        self._reconnect_attempts = 0
        self._lock = threading.Lock()
        self._reconnect_thread: Optional[threading.Thread] = None
        self._shutdown = False
        
    @property
    def connected(self) -> bool:
        """Check if connected to robot."""
        return self._connected
    
    @property
    def reconnecting(self) -> bool:
        """Check if currently attempting to reconnect."""
        return self._reconnecting
    
    @property
    def status(self) -> str:
        """Get current connection status string."""
        if self._connected:
            return "connected"
        elif self._reconnecting:
            return f"reconnecting ({self._reconnect_attempts})"
        else:
            return "disconnected"
    
    def _update_status(self, status: str) -> None:
        """Notify status change via callback."""
        if self.on_status_change:
            try:
                self.on_status_change(status)
            except Exception:
                pass
    
    def connect(self) -> bool:
        """
        Establish connection to robot.
        
        Returns:
            True if connection successful
        """
        with self._lock:
            if self._connected:
                return True
                
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.timeout)
                self._socket.connect((self.ip, self.port))
                self._connected = True
                self._reconnecting = False
                self._reconnect_attempts = 0
                self._update_status("connected")
                print(f"[Robot] Connected to {self.ip}:{self.port}")
                return True
            except socket.error as e:
                print(f"[Robot] Connection failed: {e}")
                self._connected = False
                if self._socket:
                    try:
                        self._socket.close()
                    except:
                        pass
                    self._socket = None
                self._update_status("disconnected")
                return False
    
    def disconnect(self) -> None:
        """Close connection to robot."""
        self._shutdown = True
        with self._lock:
            self._connected = False
            self._reconnecting = False
            if self._socket:
                try:
                    self._socket.close()
                except:
                    pass
                self._socket = None
            self._update_status("disconnected")
            print("[Robot] Disconnected")
        
        # Wait for reconnect thread to finish
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            self._reconnect_thread.join(timeout=1.0)
    
    def _reconnect_worker(self) -> None:
        """Background worker for automatic reconnection."""
        while not self._shutdown:
            if self._connected or not self.auto_reconnect:
                break
                
            # Check max attempts
            if (self.max_reconnect_attempts > 0 and 
                self._reconnect_attempts >= self.max_reconnect_attempts):
                print(f"[Robot] Max reconnect attempts ({self.max_reconnect_attempts}) reached")
                self._reconnecting = False
                self._update_status("disconnected")
                break
            
            self._reconnect_attempts += 1
            print(f"[Robot] Reconnect attempt {self._reconnect_attempts}...")
            self._update_status(f"reconnecting ({self._reconnect_attempts})")
            
            if self.connect():
                break
            
            # Wait before next attempt
            for _ in range(int(self.reconnect_interval * 10)):
                if self._shutdown:
                    break
                time.sleep(0.1)
        
        self._reconnecting = False
    
    def _start_reconnect(self) -> None:
        """Start background reconnection if enabled."""
        if not self.auto_reconnect or self._reconnecting:
            return
            
        self._reconnecting = True
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_worker,
            daemon=True
        )
        self._reconnect_thread.start()
    
    def send_action(self, action_name: str) -> bool:
        """
        Send action signal to robot.
        
        Args:
            action_name: Action name in Chinese (e.g., "刺拳_左")
            
        Returns:
            True if signal sent successfully
        """
        signal = ACTION_SIGNALS.get(action_name)
        if signal is None:
            if action_name in ACTION_SIGNALS:
                # Action exists but no signal defined (e.g., initial state)
                return True
            print(f"[Robot] Unknown action: {action_name}")
            return False
        
        return self.send_signal(signal)
    
    def send_signal(self, signal: str) -> bool:
        """
        Send raw signal string to robot.
        
        Args:
            signal: Signal string to send (e.g., "JAB_LEFT")
            
        Returns:
            True if signal sent successfully
        """
        with self._lock:
            if not self._connected or not self._socket:
                print(f"[Robot] Not connected, cannot send: {signal}")
                self._start_reconnect()
                return False
            
            try:
                message = f"{signal}\n"
                self._socket.sendall(message.encode('utf-8'))
                print(f"[Robot] Sent: {signal}")
                return True
            except socket.error as e:
                print(f"[Robot] Send failed: {e}")
                self._connected = False
                try:
                    self._socket.close()
                except:
                    pass
                self._socket = None
                self._update_status("disconnected")
                self._start_reconnect()
                return False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
