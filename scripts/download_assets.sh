#!/usr/bin/env bash
# Download the model file needed by the face detector (YuNet, OpenCV Zoo).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Downloading YuNet face detection model (ONNX, 232 KB)..."
curl -sL -o models/face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

echo "Done."
