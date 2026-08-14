"""Face detection with YuNet (OpenCV FaceDetectorYN) at configurable input sizes.

Why YuNet? It is the current standard face detector in OpenCV 5.x (legacy Haar
cascades and Caffe DNN backends were removed from the library).

The input resolution is the key engineering lever:
    - small input  -> fast but may miss small/far faces
    - large input  -> more sensitive, slower
The benchmark quantifies this trade-off.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
YUNET_ONNX = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

SUPPORTED_SIZES = [(160, 160), (320, 320), (640, 640)]


@dataclass
class DetectedFace:
    x: int
    y: int
    w: int
    h: int
    confidence: float


class YuNetDetector:
    """YuNet face detector with a fixed internal input resolution.

    The frame is genuinely resized to `input_size` before inference (this is the
    speed/accuracy lever), and boxes are scaled back to original coordinates.
    """

    def __init__(self, input_size=(320, 320), conf_threshold: float = 0.6):
        if not YUNET_ONNX.exists():
            raise FileNotFoundError(
                f"{YUNET_ONNX} missing — run scripts/download_assets.sh first"
            )
        self.input_size = tuple(input_size)
        self._detector = cv2.FaceDetectorYN.create(
            str(YUNET_ONNX), "", self.input_size, conf_threshold, 0.3, 5000
        )

    @property
    def name(self):
        return f"YuNet {self.input_size[0]}x{self.input_size[1]}"

    def detect(self, frame) -> list[DetectedFace]:
        ih, iw = frame.shape[:2]
        target_w, target_h = self.input_size

        # genuinely shrink/expand the frame to the internal resolution
        small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        self._detector.setInputSize(self.input_size)
        _, faces = self._detector.detect(small)

        # scale boxes back to the original frame coordinates
        sx, sy = iw / target_w, ih / target_h
        out = []
        if faces is not None and len(faces) > 0:
            for f in faces:
                x, y, w_, h_ = f[0:4]
                out.append(
                    DetectedFace(
                        int(x * sx), int(y * sy), int(w_ * sx), int(h_ * sy), float(f[14])
                    )
                )
        return out


def draw_faces(frame, faces, color=(0, 120, 255), show_confidence: bool = False):
    """Annotate a frame with bounding boxes (+ confidence)."""
    out = frame.copy()
    for f in faces:
        cv2.rectangle(out, (f.x, f.y), (f.x + f.w, f.y + f.h), color, 2)
        if show_confidence:
            label = f"{f.confidence:.2f}"
            cv2.putText(out, label, (f.x, max(10, f.y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
