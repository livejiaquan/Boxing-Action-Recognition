#!/usr/bin/env python3
"""
Test script for robot connection.

Simple server that simulates a robot controller for testing
the boxing action recognition system's robot communication.

Usage:
    python tools/test_robot.py [--port PORT]
"""

import socket
import argparse


def main():
    parser = argparse.ArgumentParser(description='Test robot server')
    parser.add_argument('--port', type=int, default=50007, help='Port to listen on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    print("=" * 50)
    print("Robot Test Server")
    print("=" * 50)
    print(f"\nListening on {args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")
    
    # Action mapping for display
    action_names = {
        "1": "刺拳_左 (Left Jab)",
        "2": "刺拳_右 (Right Jab)",
        "3": "擺拳_左 (Left Hook)",
        "4": "擺拳_右 (Right Hook)",
        "5": "上勾拳_左 (Left Uppercut)",
        "6": "上勾拳_右 (Right Uppercut)",
        "7": "防禦姿態 (Guard)",
        "8": "蹲下 (Crouch)",
        "9": "轉體_左 (Left Pivot)",
        "10": "轉體_右 (Right Pivot)",
    }
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind((args.host, args.port))
        server.listen(1)
        
        while True:
            print("[INFO] Waiting for connection...")
            conn, addr = server.accept()
            print(f"[OK] Connected: {addr}")
            
            try:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        print("[INFO] Client disconnected")
                        break
                    
                    message = data.decode('utf-8').strip()
                    action_name = action_names.get(message, "Unknown")
                    print(f"[RECV] Signal: {message} -> {action_name}")
            except ConnectionResetError:
                print("[INFO] Connection reset by client")
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        server.close()
        print("[OK] Server stopped")


if __name__ == "__main__":
    main()
