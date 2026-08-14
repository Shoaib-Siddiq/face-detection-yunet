"""Real-time webcam face detection with FPS overlay (YuNet).

Run:  python scripts/webcam_detect.py --size 320
Press q / ESC to quit.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors import YuNetDetector, draw_faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=[160, 320, 640], default=320,
                        help="YuNet internal input size (larger = more sensitive, slower)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    detector = YuNetDetector(input_size=(args.size, args.size))
    print(f"Using: {detector.name} | press 'q'/'ESC' to quit")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("ERROR: cannot open webcam")

    prev = time.perf_counter()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = detector.detect(frame)

        # FPS (exponential moving average for a stable readout)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        annotated = draw_faces(frame, faces, show_confidence=True)
        cv2.putText(annotated, f"{detector.name} | {len(faces)} face(s) | {fps:.0f} FPS",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

