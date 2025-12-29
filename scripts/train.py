#!/usr/bin/env python3
"""
Train the boxing action classifier.

This script trains a Random Forest classifier on pose keypoint features
extracted from labeled training images.

Usage:
    python scripts/train.py [--data PATH] [--output PATH]
    
Options:
    --data      Path to keypoints JSON file (default: data/keypoints_data.json)
    --output    Path to save classifier (default: models/boxing_classifier.pkl)
"""

import os
import sys
import json
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASS_NAMES, CLASS_ORDER
from src.features import extract_features, normalize_keypoints


def load_training_data(data_path: str) -> tuple:
    """
    Load and process training data from JSON file.
    
    Args:
        data_path: Path to keypoints JSON file
        
    Returns:
        Tuple of (features, labels, class_names)
    """
    print(f"\n[INFO] Loading training data from: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    features = []
    labels = []
    
    for entry in data:
        keypoints = np.array(entry['keypoints'])
        
        # Normalize keypoints
        keypoints_norm = normalize_keypoints(keypoints)
        
        # Extract features
        feature_vector = extract_features(keypoints_norm)
        features.append(feature_vector)
        labels.append(entry['class'])
    
    print(f"[OK] Loaded {len(features)} samples")
    
    # Count samples per class
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("\n[INFO] Class distribution:")
    for cls in sorted(class_counts.keys()):
        display_name = CLASS_NAMES.get(cls, cls)
        print(f"  - {cls}: {class_counts[cls]} samples ({display_name})")
    
    return np.array(features), np.array(labels), class_counts


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42
) -> tuple:
    """
    Train Random Forest classifier with cross-validation.
    
    Args:
        features: Feature matrix
        labels: Label array
        n_estimators: Number of trees
        random_state: Random seed
        
    Returns:
        Tuple of (classifier, cv_scores)
    """
    print(f"\n[INFO] Training Random Forest classifier...")
    print(f"  - Trees: {n_estimators}")
    print(f"  - Features: {features.shape[1]} dimensions")
    
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Cross-validation
    print("\n[INFO] Running 5-fold cross-validation...")
    cv_scores = cross_val_score(classifier, features, labels, cv=5)
    print(f"[OK] CV Accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")
    
    # Train final model on all data
    classifier.fit(features, labels)
    print("[OK] Classifier trained successfully")
    
    return classifier, cv_scores


def evaluate_classifier(
    classifier,
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: str
) -> None:
    """
    Evaluate classifier and save confusion matrix.
    
    Args:
        classifier: Trained classifier
        features: Feature matrix
        labels: Label array
        output_dir: Directory to save confusion matrix
    """
    # Predictions
    predictions = classifier.predict(features)
    
    # Classification report
    print("\n[INFO] Classification Report:")
    print(classification_report(labels, predictions))
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    classes = sorted(list(set(labels)))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"[OK] Confusion matrix saved to: {cm_path}")
    plt.close()


def save_model(
    classifier,
    class_names: dict,
    output_path: str,
    cv_scores: np.ndarray
) -> None:
    """
    Save trained model to file.
    
    Args:
        classifier: Trained classifier
        class_names: Class name mapping
        output_path: Output file path
        cv_scores: Cross-validation scores
    """
    model_data = {
        'classifier': classifier,
        'class_names': class_names,
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_dim': 46,
        'n_estimators': classifier.n_estimators
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    joblib.dump(model_data, output_path)
    print(f"\n[OK] Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train boxing action classifier')
    parser.add_argument(
        '--data',
        default='data/keypoints_data.json',
        help='Path to keypoints JSON file'
    )
    parser.add_argument(
        '--output',
        default='models/boxing_classifier.pkl',
        help='Path to save classifier'
    )
    parser.add_argument(
        '--trees',
        type=int,
        default=200,
        help='Number of trees in Random Forest'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Boxing Action Classifier Training")
    print("=" * 60)
    
    # Check input file
    if not os.path.exists(args.data):
        print(f"[ERROR] Data file not found: {args.data}")
        print("Please collect training data first using: python scripts/collect_data.py")
        return
    
    # Load data
    features, labels, class_counts = load_training_data(args.data)
    
    # Train classifier
    classifier, cv_scores = train_classifier(
        features, labels,
        n_estimators=args.trees
    )
    
    # Evaluate
    output_dir = os.path.dirname(args.output) or '.'
    evaluate_classifier(classifier, features, labels, output_dir)
    
    # Save model
    save_model(classifier, CLASS_NAMES, args.output, cv_scores)
    
    print("\n" + "=" * 60)
    print("[OK] Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
