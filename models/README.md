# Models Directory

This directory contains trained models for the boxing action recognition system.

## Files

- `boxing_classifier.pkl` - Pre-trained Random Forest classifier for action recognition
- `best.onnx` - (Optional) Pruned YOLO pose model in ONNX format
- `yolo11n-pose.pt` - YOLO pose model (auto-downloads on first run)

## Download Instructions

The YOLO pose model (`yolo11n-pose.pt`) will be automatically downloaded when you first run the system. Alternatively, you can manually download it from [Ultralytics](https://docs.ultralytics.com/).

## Training Your Own Classifier

To train a new classifier with your own data:

```bash
python scripts/train.py --data data/keypoints_data.json --output models/boxing_classifier.pkl
```

See `scripts/train.py` for more options.
