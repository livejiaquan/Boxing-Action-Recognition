# Boxing Action Recognition

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLO](https://img.shields.io/badge/YOLO-v11--pose-green.svg)](https://docs.ultralytics.com/)

Real-time boxing action recognition system using **YOLO11-pose** for pose estimation and **Random Forest** for action classification. Supports multi-person detection, automatic target selection, and optional robot integration via TCP socket.

<p align="center">
  <img src="docs/images/demo.gif" alt="Demo" width="600">
</p>

## ✨ Features

- **11 Boxing Actions**: Recognizes stances, punches, pivots, and defensive moves
- **Real-time Performance**: 25-30 FPS on modern hardware
- **Multi-person Detection**: Automatic target selection based on proximity and confidence
- **Frame Confirmation**: Reduces false positives with consecutive frame validation
- **Robot Integration**: TCP socket interface for robot control (optional)
- **Auto-reconnect**: Automatic reconnection for robot communication

## 🎯 Supported Actions

| Action | Chinese Name | Signal |
|--------|-------------|--------|
| Initial Stance | 初始狀態 | - |
| Guard Position | 防禦姿態 | 7 |
| Crouch/Duck | 蹲下 | 8 |
| Left Pivot | 轉體_左 | 9 |
| Right Pivot | 轉體_右 | 10 |
| Left Jab | 刺拳_左 | 1 |
| Right Jab | 刺拳_右 | 2 |
| Left Hook | 擺拳_左 | 3 |
| Right Hook | 擺拳_右 | 4 |
| Left Uppercut | 上勾拳_左 | 5 |
| Right Uppercut | 上勾拳_右 | 6 |

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Webcam Input  │ --> │  YOLO11-pose     │ --> │ 17 Keypoints    │
│   (640x480)     │     │  Pose Estimation │     │ (x, y, conf)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Robot Control  │ <-- │  Random Forest   │ <-- │ 46-dim Features │
│  (TCP Socket)   │     │  Classifier      │     │ (normalized)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Feature Extraction (46 dimensions)

- Joint coordinates (24d) - Shoulder, elbow, wrist, hip, knee, ankle
- Wrist absolute position (4d)
- Arm extension ratios (2d)
- Wrist-shoulder displacement (4d)
- Bilateral wrist differences (3d)
- Shoulder rotation angle (1d)
- Elbow joint angles (2d)
- Knee joint angles (2d)
- Hip center position (2d)
- Wrist-to-hip distances (2d)

## 📦 Installation

### Prerequisites

- Python 3.10+
- Webcam
- (Optional) NVIDIA GPU with CUDA for faster inference

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/boxing-action-recognition.git
cd boxing-action-recognition

# Create virtual environment (recommended: conda)
conda create -n boxing python=3.10 -y
conda activate boxing

# Or use venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (auto-downloads on first run)
# Or manually place yolo11n-pose.pt in models/
```

## 🚀 Quick Start

### Run Recognition System

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `SPACE` | Pause/Resume recognition |

## 📊 Training Custom Models

### 1. Collect Training Data

```bash
python scripts/collect_data.py
```

Press `0-9` to label poses with corresponding action classes.

### 2. Train Classifier

```bash
python scripts/train.py --data data/keypoints_data.json --output models/boxing_classifier.pkl
```

## ⚙️ Configuration

Edit `config.py` to customize behavior:

```python
# Recognition Parameters
CONFIDENCE_THRESHOLD = 0.50  # Action confidence threshold
CONFIRM_FRAMES = 10          # Frames required for confirmation
COOLDOWN_TIME = 3.0          # Seconds between repeated actions

# Robot Interface
ROBOT_ENABLED = True         # Enable robot communication
ROBOT_IP = "192.168.1.95"    # Robot IP address
ROBOT_PORT = 50007           # Robot port
```

## 📁 Project Structure

```
boxing-action-recognition/
├── main.py                 # Main entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── src/                    # Core modules
│   ├── __init__.py
│   ├── features.py         # Feature extraction
│   ├── pose_validator.py   # Body completeness checks
│   ├── robot_client.py     # Robot TCP communication
│   └── visualizer.py       # UI rendering
├── scripts/                # Training utilities
│   ├── train.py            # Train classifier
│   └── collect_data.py     # Collect training data
├── models/                 # Model files
│   └── boxing_classifier.pkl
├── data/                   # Training data
│   └── keypoints_data.json
└── docs/                   # Documentation
    └── images/
```

## 🔧 Robot Integration

The system sends action signals via TCP socket:

```python
# Signal format: "{action_number}\n"
# Example: "1\n" for left jab

# To receive in your robot controller:
import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 50007))
server.listen(1)

conn, addr = server.accept()
while True:
    data = conn.recv(1024).decode('utf-8')
    if data:
        action = data.strip()
        # Process action...
```

## 📈 Performance

| Hardware | FPS | Notes |
|----------|-----|-------|
| MacBook Pro M4 | ~30 | CPU inference |
| Desktop GTX 3080 | ~50+ | GPU inference |
| Raspberry Pi 4 | ~5-8 | CPU only |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for pose estimation
- [scikit-learn](https://scikit-learn.org/) for Random Forest implementation
- COCO dataset for keypoint definitions

---

<p align="center">
  Made with ❤️ for robotics and computer vision
</p>
